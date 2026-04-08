import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def build_robot_description(context: LaunchContext, arm_id, load_gripper, franka_hand,
                            lock_gripper_closed, tcp_xyz):
    arm_id_str = context.perform_substitution(arm_id)
    load_gripper_str = context.perform_substitution(load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)
    lock_gripper_closed_str = context.perform_substitution(lock_gripper_closed)
    tcp_xyz_str = context.perform_substitution(tcp_xyz)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        arm_id_str,
        arm_id_str + '.urdf.xacro',
    )
    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'arm_id': arm_id_str,
            'hand': load_gripper_str,
            'ros2_control': 'true',
            'gazebo': 'true',
            'ee_id': franka_hand_str,
            'gazebo_effort': 'true',
            'lock_gripper_closed': lock_gripper_closed_str,
            'tcp_xyz': tcp_xyz_str,
        },
    )
    robot_description_xml = robot_description_config.toxml()

    urdf_path = f'/tmp/{arm_id_str}.urdf'
    with open(urdf_path, 'w', encoding='utf-8') as urdf_file:
        urdf_file.write(robot_description_xml)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description_xml}],
    )
    return [robot_state_publisher]


def generate_launch_description():
    arm_id = LaunchConfiguration('arm_id')
    load_gripper = LaunchConfiguration('load_gripper')
    franka_hand = LaunchConfiguration('franka_hand')
    lock_gripper_closed = LaunchConfiguration('lock_gripper_closed')
    tcp_xyz = LaunchConfiguration('tcp_xyz')
    start_rviz = LaunchConfiguration('start_rviz')
    spawn_hemisphere = LaunchConfiguration('spawn_hemisphere')
    params_file = LaunchConfiguration('params_file')

    rviz_config_default = PathJoinSubstitution(
        [FindPackageShare('franka_surface_impedance_controller'), 'config', 'surface_debug_markers.rviz']
    )
    params_file_default = PathJoinSubstitution(
        [FindPackageShare('franka_surface_impedance_controller'), 'config', 'surface_impedance_controller.yaml']
    )

    robot_state_publisher = OpaqueFunction(
        function=build_robot_description,
        args=[arm_id, load_gripper, franka_hand, lock_gripper_closed, tcp_xyz],
    )

    gazebo_empty_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': 'empty.sdf -r'}.items(),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description'],
        output='screen',
    )

    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'source_list': ['joint_states'], 'rate': 30}],
        output='screen',
    )

    surface_scan_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('franka_surface_impedance_controller'),
                 'launch', 'spawn_surface_impedance_controller.launch.py']
            )
        ),
        launch_arguments={
            'controller_manager': '/controller_manager',
            'params_file': params_file,
            'start_rviz': start_rviz,
            'rviz_config': rviz_config_default,
            'spawn_hemisphere': spawn_hemisphere,
            'spawn_joint_state_broadcaster': 'false',
            'spawn_franka_robot_state_broadcaster': 'false',
            'spawn_surface_inactive': 'false',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('arm_id', default_value='fer'),
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument('franka_hand', default_value='franka_hand'),
        DeclareLaunchArgument('lock_gripper_closed', default_value='true'),
        DeclareLaunchArgument(
            'tcp_xyz',
            default_value='0 0 0.115',
            description='Scan-specific virtual TCP placed ahead of the closed finger tips with extra shell-clearance margin',
        ),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('spawn_hemisphere', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=params_file_default),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(get_package_share_directory('franka_description')),
        ),
        gazebo_empty_world,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_robot,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[surface_scan_launch],
            )
        ),
    ])
