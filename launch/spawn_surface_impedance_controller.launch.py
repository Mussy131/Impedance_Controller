import math

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_hemisphere_center_from_params(params_file_path):
    default_center = ('0.65', '0.0', '0.02')
    try:
        with open(params_file_path, 'r', encoding='utf-8') as params_file:
            params = yaml.safe_load(params_file) or {}
        center = (
            params
            .get('surface_impedance_controller', {})
            .get('ros__parameters', {})
            .get('surface', {})
            .get('hemisphere', {})
            .get('center')
        )
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            return default_center
        resolved = []
        for value in center:
            numeric = float(value)
            if not math.isfinite(numeric):
                return default_center
            resolved.append(str(numeric))
        return tuple(resolved)
    except Exception:
        return default_center


def _spawn_hemisphere_from_launch_args(context, params_file, spawn_hemisphere, hemisphere_world,
                                       hemisphere_name, hemisphere_model,
                                       hemisphere_x, hemisphere_y, hemisphere_z):
    resolved_xyz = [
        context.perform_substitution(hemisphere_x).strip(),
        context.perform_substitution(hemisphere_y).strip(),
        context.perform_substitution(hemisphere_z).strip(),
    ]
    params_xyz = _load_hemisphere_center_from_params(
        context.perform_substitution(params_file)
    )
    resolved_xyz = [
        params_value if value in ('', 'from_params') else value
        for value, params_value in zip(resolved_xyz, params_xyz)
    ]

    return [
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-world', context.perform_substitution(hemisphere_world),
                '-name', context.perform_substitution(hemisphere_name),
                '-file', context.perform_substitution(hemisphere_model),
                '-x', resolved_xyz[0],
                '-y', resolved_xyz[1],
                '-z', resolved_xyz[2],
            ],
            output='screen',
            condition=IfCondition(spawn_hemisphere),
        )
    ]


def generate_launch_description():
    controller_manager = LaunchConfiguration('controller_manager')
    params_file_default = PathJoinSubstitution(
        [FindPackageShare('franka_surface_impedance_controller'), 'config', 'surface_impedance_controller.yaml']
    )
    rviz_config_default = PathJoinSubstitution(
        [FindPackageShare('franka_surface_impedance_controller'), 'config', 'surface_debug_markers.rviz']
    )
    hemi_model_default = PathJoinSubstitution(
        [FindPackageShare('franka_surface_impedance_controller'), 'models', 'hemi_surface', 'model.sdf']
    )
    params_file = LaunchConfiguration('params_file')
    start_rviz = LaunchConfiguration('start_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    spawn_hemisphere = LaunchConfiguration('spawn_hemisphere')
    hemisphere_world = LaunchConfiguration('hemisphere_world')
    hemisphere_name = LaunchConfiguration('hemisphere_name')
    hemisphere_model = LaunchConfiguration('hemisphere_model')
    hemisphere_x = LaunchConfiguration('hemisphere_x')
    hemisphere_y = LaunchConfiguration('hemisphere_y')
    hemisphere_z = LaunchConfiguration('hemisphere_z')
    spawn_joint_state_broadcaster = LaunchConfiguration('spawn_joint_state_broadcaster')
    spawn_franka_robot_state_broadcaster = LaunchConfiguration('spawn_franka_robot_state_broadcaster')
    spawn_surface_inactive = LaunchConfiguration('spawn_surface_inactive')

    return LaunchDescription([
        DeclareLaunchArgument('controller_manager', default_value='/controller_manager'),
        DeclareLaunchArgument('params_file', default_value=params_file_default),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=rviz_config_default),
        DeclareLaunchArgument('spawn_hemisphere', default_value='true'),
        DeclareLaunchArgument('hemisphere_world', default_value='empty'),
        DeclareLaunchArgument('hemisphere_name', default_value='hemi_surface'),
        DeclareLaunchArgument('hemisphere_model', default_value=hemi_model_default),
        DeclareLaunchArgument(
            'hemisphere_x',
            default_value='from_params',
            description='Hemisphere world X. Use "from_params" to read surface.hemisphere.center from params_file.',
        ),
        DeclareLaunchArgument(
            'hemisphere_y',
            default_value='from_params',
            description='Hemisphere world Y. Use "from_params" to read surface.hemisphere.center from params_file.',
        ),
        DeclareLaunchArgument(
            'hemisphere_z',
            default_value='from_params',
            description='Hemisphere world Z. Use "from_params" to read surface.hemisphere.center from params_file.',
        ),
        DeclareLaunchArgument('spawn_joint_state_broadcaster', default_value='false'),
        DeclareLaunchArgument('spawn_franka_robot_state_broadcaster', default_value='false'),
        DeclareLaunchArgument('spawn_surface_inactive', default_value='true'),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', controller_manager],
            output='screen',
            condition=IfCondition(spawn_joint_state_broadcaster),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['franka_robot_state_broadcaster', '--controller-manager', controller_manager],
            output='screen',
            condition=IfCondition(spawn_franka_robot_state_broadcaster),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'surface_impedance_controller',
                '--controller-manager',
                controller_manager,
                '--param-file',
                params_file,
                '--inactive',
            ],
            output='screen',
            condition=IfCondition(spawn_surface_inactive),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'surface_impedance_controller',
                '--controller-manager',
                controller_manager,
                '--param-file',
                params_file,
            ],
            output='screen',
            condition=UnlessCondition(spawn_surface_inactive),
        ),
        OpaqueFunction(
            function=_spawn_hemisphere_from_launch_args,
            args=[
                params_file,
                spawn_hemisphere,
                hemisphere_world,
                hemisphere_name,
                hemisphere_model,
                hemisphere_x,
                hemisphere_y,
                hemisphere_z,
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(start_rviz),
        ),
    ])
