# Copyright (c) 2025 Testing Automated group (TAU) at 
# the università della svizzera italiana (USI), Switzerland
#
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0

import gc
import importlib
import logging
import os
import sys

# CaRL enables deterministic CUDA algorithms at import time. CuBLAS requires
# this process-level setting before the first CUDA operation.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

# Ensure imports work regardless of caller's working directory
pcla_dir = os.path.dirname(os.path.abspath(__file__))
if pcla_dir not in sys.path:
    sys.path.insert(0, pcla_dir)

# Add lmdrive's custom timm (vision_encoder) before anything that might import timm
lmdrive_vision_encoder = os.path.join(pcla_dir, 'pcla_agents', 'lmdrive', 'vision_encoder')
if os.path.exists(lmdrive_vision_encoder) and lmdrive_vision_encoder not in sys.path:
    sys.path.insert(0, lmdrive_vision_encoder)

import carla
from pcla_functions import give_path, setup_sensor_attributes, location_to_waypoint, route_maker
from leaderboard_codes.watchdog import Watchdog
from leaderboard_codes.timer import GameTime
from leaderboard_codes.route_indexer import RouteIndexer
from leaderboard_codes.carla_data_provider import CarlaDataProvider
from leaderboard_codes.route_manipulation import interpolate_trajectory
from leaderboard_codes.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader

logger = logging.getLogger(__name__)


def _reset_torch_runtime_state():
    """Restore process-global PyTorch settings before switching agents."""
    try:
        import torch
    except ImportError:
        return

    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


class PCLA():
    def __init__(self, agent, vehicle, route, client, destroy_vehicle=False):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.client = None
        self.world = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.agent_instance = None
        self.routePath = None
        self._watchdog = None
        self._sensors = []
        self._destroy_vehicle = destroy_vehicle
        self._observation_proxy = None
        self.set(agent, vehicle, route, client)
    
    def set(self, agent, vehicle, route, client):
        self.client = client
        self.world = client.get_world()
        self.vehicle = vehicle
        self.routePath = route
        self._watchdog = Watchdog(260) # TODO: Increase timeout if needed for large models
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        try:
            self.setup_agent(agent)
            self.setup_route()
            self.setup_sensors()
        except Exception:
            self.cleanup()
            raise

    def setup_agent(self, agent):
        _reset_torch_runtime_state()
        GameTime.restart()
        self._watchdog.start()
        self.agentPath, self.configPath = give_path(agent, self.current_dir, self.routePath)
        # Reload the agent freshly each time to avoid stale imports across agents.
        # Use file-based loading to prevent cross-talk between agents sharing module names (e.g., model.py).
        module_name = os.path.basename(self.agentPath).split('.')[0]
        module_dir = os.path.dirname(self.agentPath)
        module_key = f"pcla_dynamic_agent.{module_name}"

        # Prepend the agent directory so its relative imports resolve to local files.
        # Save/restore sys.path to isolate dependencies per agent.
        original_sys_path = list(sys.path)
        if module_dir in sys.path:
            sys.path.remove(module_dir)
        sys.path.insert(0, module_dir)

        # Drop previously loaded local agent modules to avoid cross-agent contamination
        # (e.g., plant2's dataset being reused by plant1).
        module_names_to_clear = {
            module_key,
            'model', 'models', 'dataset', 'lit_module', 'plant_variables',
            'nav_planner', 'config', 'transfuser', 'transfuser_utils',
            'utils', 'util', 'data', 'gaussian_target',
            'planner', 'controller', 'map_agent', 'base_agent',
            'bev_planner', 'waypointer', 'lateral_controller',
            'longitudinal_controller', 'kinematic_bicycle_model'
        }
        module_prefixes_to_clear = (
            'models.',
            'util.',
            'carla_garage.',
            'birds_eye_view.',
        )
        for key in list(sys.modules.keys()):
            if key in module_names_to_clear or key == 'carla_garage' or key.startswith(module_prefixes_to_clear):
                del sys.modules[key]

        try:
            spec = importlib.util.spec_from_file_location(module_key, self.agentPath)
            module_agent = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module_agent
            spec.loader.exec_module(module_agent)
        finally:
            # Restore sys.path
            sys.path = original_sys_path
        
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        self.agent_instance = getattr(module_agent, agent_class_name)(self.configPath)

        self._watchdog.stop()

    def setup_route(self):
        scenarios = os.path.join(self.current_dir, "leaderboard_codes/no_scenarios.json")
        route_indexer = RouteIndexer(self.routePath, scenarios, 1)
        config = route_indexer.next()
        
        gps_route, route = interpolate_trajectory(self.world, config.trajectory)

        self.agent_instance.set_global_plan(gps_route, route)

    def setup_sensors(self):
        """Attach sensors defined by the agent to the ego-vehicle."""
        bp_library = self.world.get_blueprint_library()
        try:
            for sensor_spec in self.agent_instance.sensors():
                # Pseudosensors (not spawned)
                if sensor_spec['type'].startswith('sensor.opendrive_map'):
                    sensor = OpenDriveMapReader(self.vehicle, sensor_spec['reading_frequency'])
                elif sensor_spec['type'].startswith('sensor.speedometer'):
                    delta_time = 1/20
                    frame_rate = 1 / delta_time
                    sensor = SpeedometerReader(self.vehicle, frame_rate)
                else:
                    # World sensors (spawned actors)
                    bp = bp_library.find(str(sensor_spec['type']))
                    bp_setup = setup_sensor_attributes(bp, sensor_spec)
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z'])
                    if sensor_spec['type'].startswith('sensor.other.gnss'):
                        sensor_rotation = carla.Rotation()
                    else:
                        sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'], roll=sensor_spec['roll'], yaw=sensor_spec['yaw'])

                    # Create sensor actor
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self.world.spawn_actor(bp_setup, sensor_transform, self.vehicle)
                self._sensors.append(sensor)
                sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.agent_instance.sensor_interface))
        except Exception:
            self._cleanup_sensors()
            raise

        CarlaDataProvider.register_actor(self.vehicle)
            
    def get_action(self, snapshot=None):
        if snapshot is None:
            snapshot = self.world.get_snapshot()
        timestamp = None
        if snapshot:
            timestamp = snapshot.timestamp
        if timestamp:
            GameTime.on_carla_tick(timestamp)
            decision_vehicle = self.vehicle
            if getattr(self, "_observation_proxy", None) is not None:
                decision_vehicle = self._observation_proxy.proxy_for_actor(
                    self.vehicle,
                    world=self.world,
                )
            return self.agent_instance(vehicle=decision_vehicle)

    def configure_observation_proxy(self, registry):
        """Install observation-first reads for the agent decision boundary only."""
        self._observation_proxy = registry
        setter = getattr(CarlaDataProvider, "set_observation_registry", None)
        if callable(setter):
            setter(registry, world=self.world)
        sensor_interface = getattr(self.agent_instance, "sensor_interface", None)
        setter = getattr(sensor_interface, "set_observation_proxy", None)
        if callable(setter):
            setter(registry, self.vehicle)

    def done(self):
        if self.agent_instance is None:
            return False
        for method_name in ("done", "is_done"):
            method = getattr(self.agent_instance, method_name, None)
            if callable(method):
                return bool(method())
        return False

    def _cleanup_sensors(self):
        for sensor in reversed(self._sensors):
            try:
                is_listening = getattr(sensor, "is_listening", None)
                if callable(is_listening):
                    is_listening = is_listening()
                if is_listening and hasattr(sensor, "stop"):
                    sensor.stop()
            except Exception:
                logger.exception("Failed to stop PCLA-owned sensor")
            try:
                if hasattr(sensor, "destroy"):
                    sensor.destroy()
            except Exception:
                logger.exception("Failed to destroy PCLA-owned sensor")
        self._sensors.clear()

    def _clear_data_provider_state(self):
        """Clear provider bookkeeping without destroying actors owned elsewhere."""
        for attribute in (
            "_actor_velocity_map",
            "_actor_location_map",
            "_actor_transform_map",
            "_actor_refs",
            "_traffic_light_map",
            "_carla_actor_pool",
            "_vehicles_with_open_doors",
        ):
            value = getattr(CarlaDataProvider, attribute, None)
            if hasattr(value, "clear"):
                value.clear()
        for attribute in (
            "_map",
            "_world",
            "_all_actors",
            "_client",
            "_spawn_points",
            "_ego_vehicle_route",
            "_grp",
        ):
            if hasattr(CarlaDataProvider, attribute):
                setattr(CarlaDataProvider, attribute, None)
        if hasattr(CarlaDataProvider, "_sync_flag"):
            CarlaDataProvider._sync_flag = False
        if hasattr(CarlaDataProvider, "_spawn_index"):
            CarlaDataProvider._spawn_index = 0
        if hasattr(CarlaDataProvider, "_runtime_init_flag"):
            CarlaDataProvider._runtime_init_flag = False
        if hasattr(CarlaDataProvider, "_observation_registry"):
            CarlaDataProvider._observation_registry = None
        if hasattr(CarlaDataProvider, "active_scenarios"):
            CarlaDataProvider.active_scenarios = []
        if hasattr(CarlaDataProvider, "last_scenario"):
            CarlaDataProvider.last_scenario = None
    
    def cleanup(self):
        """Remove and destroy all actors."""

        if self._watchdog:
            self._watchdog.stop()

        # Cleanup the agent first so it can stop any internal threads and sensors
        try:
            if self.agent_instance is not None:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception:
            logger.exception("Failed to stop the PCLA agent")

        self._cleanup_sensors()

        if getattr(self, "_observation_proxy", None) is not None:
            clearer = getattr(CarlaDataProvider, "clear_observation_registry", None)
            if callable(clearer):
                clearer(self._observation_proxy)
            self._observation_proxy = None

        if self._destroy_vehicle:
            try:
                if self.vehicle is not None and self.vehicle.is_alive:
                    self.vehicle.destroy()
            except RuntimeError:
                pass

        self.current_dir = None
        self.client = None
        self.vehicle = None
        self.agentPath = None
        self.configPath = None
        self.routePath = None
        self.world = None

        self._clear_data_provider_state()

        # Release cached CUDA memory between agents to avoid cross-agent OOMs.
        try:
            import torch

            _reset_torch_runtime_state()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
