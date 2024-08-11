#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
import tf2_ros
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import SetBool

from copy import deepcopy
import numpy as np
import math
import heapq
import open3d as o3d

from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

INCREASED_MAP_RESOLUTION = 0.05
SPEED = 0.4
LOOKAHEAD_DISTANCE = 0.4
TARGET_ERROR = 0.2
EXPANSION_SIZE = 3
Z_THRESHOLD = 0.23
GRID_RESOLUTION = 0.05
voxel_size = GRID_RESOLUTION


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def pure_pursuit(current_x, current_y, current_heading, path, index, speed, lookahead_distance, forward=True):
    closest_point = None
    if forward:
        v = speed  # Set the speed to a negative value to make the robot go in reverse
    else:
        v = -speed  # Set the speed to a negative value to make the robot go in reverse
    for i in range(index, len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        if forward:
            target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        else:
            target_heading = math.atan2(current_y - closest_point[1], current_x - closest_point[0])  # Reverse the atan2 arguments
        desired_steering_angle = target_heading - current_heading
    else:
        if forward:
            target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        else:
            target_heading = math.atan2(current_y - path[-1][1], current_x - path[-1][0])  # Reverse the atan2 arguments
        desired_steering_angle = target_heading - current_heading
        index = len(path) - 1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi / 6 or desired_steering_angle < -math.pi / 6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = (sign * math.pi / 4)
        v = 0.0
    return v, desired_steering_angle, index


def find_nearest_3d_point(x, y, z, array_3d):
    indices = np.argwhere(array_3d == 100)
    
    distances = np.sqrt((indices[:, 0] - x)**2 + (indices[:, 1] - y)**2 + (indices[:, 2] - z)**2)
    nearest_index = indices[np.argmin(distances)]
    
    return nearest_index


def astar(array, start, goal):
    neighbors = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1),  (-1, 0, 0),  (-1, 0, 1),
        (-1, 1, -1),  (-1, 1, 0),  (-1, 1, 1),
        
        (0, -1, -1),  (0, -1, 0),  (0, -1, 1),
        (0, 0, -1),               (0, 0, 1),
        (0, 1, -1),   (0, 1, 0),   (0, 1, 1),
        
        (1, -1, -1),  (1, -1, 0),  (1, -1, 1),
        (1, 0, -1),   (1, 0, 0),   (1, 0, 1),
        (1, 1, -1),   (1, 1, 0),   (1, 1, 1)
    ]

    def heuristic(a, b):
        # Euclidean distance as heuristic
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    def is_within_z_constraint(current, neighbor, z_constraint=0.2):
        z_constraint = int(z_constraint / voxel_size)
        return abs(current[2] - neighbor[2]) <= z_constraint

    def is_within_bounds(coord):
        return 0 <= coord[0] < array.shape[0] and 0 <= coord[1] < array.shape[1] and 0 <= coord[2] < array.shape[2]
    
    def is_occupied_space(coord):
        return array[coord[0], coord[1], coord[2]] == 100

    def has_no_occupied_cells_above(coord, vertical_min=0.3, vertical_range=0.6):
        z_min = coord[2] + int(vertical_min / voxel_size)  # Start checking
        z_max = coord[2] + int(vertical_range / voxel_size)  # End checking within the vertical range

        # Use list comprehension to create a list of coordinates to check
        coords_to_check = [(coord[0], coord[1], z) for z in range(z_min, z_max + 1) if is_within_bounds((coord[0], coord[1], z))]

        # Check all coordinates in one batch if possible
        return not any(is_occupied_space(c) for c in coords_to_check)

    def is_cylinder_collision_free(coord, radius):
        # Convert radius and base offset from meters to voxel grid units
        grid_radius = radius / voxel_size
        grid_z_start = int(0.3 / voxel_size)
        grid_z_end = int(0.6 / voxel_size)

        # Calculate the number of points to check around the circumference
        num_points = int(2 * math.pi * grid_radius)
        
        # Iterate over the points on the circumference
        for angle in range(0, num_points, 2):
            theta = 2 * math.pi * angle / num_points
            i = int(grid_radius * math.cos(theta))
            j = int(grid_radius * math.sin(theta))
            
            for k in range(grid_z_start, grid_z_end + 1, 2):
                check_coord = (coord[0] + i, coord[1] + j, coord[2] + k)
                if is_within_bounds(check_coord) and is_occupied_space(check_coord):
                    return False  # Early exit if any occupied space is found
        
        return True  # No collision found

    open_list = []
    heapq.heappush(open_list, (0, start))  # Priority queue with (f-score, node)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for i, j, k in neighbors:
            neighbor = (current[0] + i, current[1] + j, current[2] + k)

            # Ensure the neighbor is within the grid bounds
            if not (0 <= neighbor[0] < array.shape[0] and
                    0 <= neighbor[1] < array.shape[1] and
                    0 <= neighbor[2] < array.shape[2]):
                continue

            # Plan through occupied cells only
            if array[neighbor[0], neighbor[1], neighbor[2]] != 100:
                continue

            # Check z constraint
            if not is_within_z_constraint(current, neighbor):
                continue

            # Check if there are occupied cells above the neighbor
            if not has_no_occupied_cells_above(neighbor):
                continue

            if not is_cylinder_collision_free(neighbor, 0.3):
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # Return None if no path is found


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


class PointCloudToGrid(Node):
    def __init__(self):
        super().__init__('pointcloud_to_grid_node')

        # Parameters
        self.declare_parameter('cloud_in_topic', '/navigation/octomap_point_cloud_centers_filtered')
        self.declare_parameter('cell_size', GRID_RESOLUTION)
        self.declare_parameter('z_threshold', Z_THRESHOLD)
        self.declare_parameter('verbose', False)

        self.cloud_in_topic = self.get_parameter('cloud_in_topic').get_parameter_value().string_value
        self.cell_size = self.get_parameter('cell_size').get_parameter_value().double_value
        self.z_threshold = self.get_parameter('z_threshold').get_parameter_value().double_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Publisher and Subscriber
        self.sub_pc2 = self.create_subscription(PointCloud2, self.cloud_in_topic, self.pointcloud2_callback, 1)

        self.listen_to_pointcloud = True
        self.create_timer(1, self.timer_cb)

        # Initialize the transform buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_cb, 1)
        self.path_publisher = self.create_publisher(Marker, 'path_marker', 1)

        self.create_timer(0.1, self.path_follower_timer_callback)

        self.twist_publisher = self.create_publisher(Twist, '/cmd_vel', 1)

        self.in_motion = False
        self.srv = self.create_service(SetBool, '/navigation/follow_path', self.set_in_motion_callback)

        self.server = InteractiveMarkerServer(self, "goal_marker")
        self.create_goal_marker(0.,0.,0.)

    def set_in_motion_callback(self, request, response):
        self.in_motion = request.data
        response.success = True
        response.message = f"in_motion set to {self.in_motion}"
        return response

    def path_follower_timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform("map", "body", rclpy.time.Time())

            # Extract the robot's position and orientation in the "map" frame
            self.x = transform.transform.translation.x
            self.y = transform.transform.translation.y
            self.robot_yaw = yaw_from_quaternion(transform.transform.rotation.x,
                                                transform.transform.rotation.y,
                                                transform.transform.rotation.z,
                                                transform.transform.rotation.w)

            self.robot_position = [self.x, self.y]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            # self.get_logger().warn("Could not get transform from map to body: {}".format(ex))
            return

        if not self.in_motion:
            return

        if not hasattr(self, "current_path_world"):
            self.get_logger().warn("Path Not Yet Planned")
            return
        
        if len(self.current_path_world) == 0:
            return
        
        linear_velocity, angular_velocity, self.pursuit_index = pure_pursuit(
            self.x,
            self.y,
            self.robot_yaw,
            self.current_path_world,
            self.pursuit_index,
            SPEED,
            LOOKAHEAD_DISTANCE
        )

        if(abs(self.x - self.current_path_world[-1][0]) < TARGET_ERROR and abs(self.y - self.current_path_world[-1][1]) < TARGET_ERROR):
            self.in_motion = False
            print("Target reached")
            linear_velocity = 0
            angular_velocity = 0
            self.current_path_world = []

        # Publish the twist commands
        twist_command = Twist()
        twist_command.linear.x = float(linear_velocity)
        twist_command.angular.z = float(angular_velocity)
        self.twist_publisher.publish(twist_command)

    def create_goal_marker(self, x, y, z):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.pose.position.x = x
        int_marker.pose.position.y = y
        int_marker.pose.position.z = z
        int_marker.scale = 1.0

        # Create the marker (a simple cube here for visualization)
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Attach the marker to a control for visualization
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)

        # Add control for moving in x, y, z
        control_x = InteractiveMarkerControl()
        control_x.name = "move_x"
        control_x.orientation.w = 1.
        control_x.orientation.x = 1.
        control_x.orientation.y = 0.
        control_x.orientation.z = 0.
        control_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_x)

        control_y = InteractiveMarkerControl()
        control_y.name = "move_y"
        control_y.orientation.w = 1.
        control_y.orientation.x = 0.
        control_y.orientation.y = 1.
        control_y.orientation.z = 0.
        control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_y)

        control_z = InteractiveMarkerControl()
        control_z.name = "move_z"
        control_z.orientation.w = 1.
        control_z.orientation.x = 0.
        control_z.orientation.y = 0.
        control_z.orientation.z = 1.
        control_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_z)

        # Add control for rotating in x, y, z
        control_rot_x = InteractiveMarkerControl()
        control_rot_x.name = "rotate_x"
        control_rot_x.orientation.w = 1.
        control_rot_x.orientation.x = 1.
        control_rot_x.orientation.y = 0.
        control_rot_x.orientation.z = 0.
        control_rot_x.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_rot_x)

        control_rot_y = InteractiveMarkerControl()
        control_rot_y.name = "rotate_y"
        control_rot_y.orientation.w = 1.
        control_rot_y.orientation.x = 0.
        control_rot_y.orientation.y = 1.
        control_rot_y.orientation.z = 0.
        control_rot_y.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_rot_y)

        control_rot_z = InteractiveMarkerControl()
        control_rot_z.name = "rotate_z"
        control_rot_z.orientation.w = 1.
        control_rot_z.orientation.x = 0.
        control_rot_z.orientation.y = 0.
        control_rot_z.orientation.z = 1.
        control_rot_z.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_rot_z)

        # Add the interactive marker to the server
        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        self.goal_position = [feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z]

    def pose_cb(self, msg):
        if not hasattr(self, "robot_foot_location"):
            self.get_logger().warn(f"Where is robot?")
            return
        self.get_logger().info(f"Searching for path ...")
        
        # Define goal coordinates in the same reference frame as the point cloud
        if not hasattr(self, "goal_position"):
            self.get_logger().warn(f"Set the goal position using /goal_marker/update in rviz2")
            return
        goal_coordinates_raw = self.goal_position
        start_coordinates_raw = self.robot_foot_location

        # Offset coordinates by the minimum bounds of the point cloud
        goal_coordinates = (
            int((goal_coordinates_raw[0] - self.min_bound[0]) / GRID_RESOLUTION),
            int((goal_coordinates_raw[1] - self.min_bound[1]) / GRID_RESOLUTION),
            int((goal_coordinates_raw[2] - self.min_bound[2]) / GRID_RESOLUTION)
        )

        self.x, self.y, self.z = [
            int((start_coordinates_raw[0] - self.min_bound[0]) / GRID_RESOLUTION),
            int((start_coordinates_raw[1] - self.min_bound[1]) / GRID_RESOLUTION),
            int((start_coordinates_raw[2] - self.min_bound[2]) / GRID_RESOLUTION)
        ]

        self.x, self.y, self.z = find_nearest_3d_point(self.x, self.y, self.z, self.occupancy_grid)

        # Check if the calculated indices are within the bounds of the occupancy grid
        if 0 <= goal_coordinates[0] < self.occupancy_grid.shape[0] and \
        0 <= goal_coordinates[1] < self.occupancy_grid.shape[1] and \
        0 <= goal_coordinates[2] < self.occupancy_grid.shape[2]:
            self.occupancy_grid[goal_coordinates[0], goal_coordinates[1], goal_coordinates[2]] = 100
        else:
            self.get_logger().warn(f"Goal coordinates {goal_coordinates} are out of bounds.")

        if 0 <= self.x < self.occupancy_grid.shape[0] and \
        0 <= self.y < self.occupancy_grid.shape[1] and \
        0 <= self.z < self.occupancy_grid.shape[2]:
            self.occupancy_grid[self.x, self.y, self.z] = 100
        else:
            self.get_logger().warn(f"Start coordinates ({self.x}, {self.y}, {self.z}) are out of bounds.")
            return

        path = astar(
            self.occupancy_grid, (self.x, self.y, self.z),
            goal_coordinates
        )

        if path is None:
            self.get_logger().warn("Could not reach target")
            return
        
        self.get_logger().info("Path to the target found")

        path_marker = Marker()
        path_marker.header = msg.header
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.1  # Line width
        path_marker.color.a = 1.0  # Alpha
        path_marker.color.r = 0.0  # Red
        path_marker.color.g = 1.0  # Green
        path_marker.color.b = 0.0  # Blue

        self.current_path_world = []
        for p in path:
            pose = Pose()

            # Convert from grid coordinates (p[0], p[1]) to world coordinates
            pose.position.x, pose.position.y, pose.position.z = (p[0])* GRID_RESOLUTION + self.min_bound[0], (p[1]) * GRID_RESOLUTION + self.min_bound[1], (p[2]) * GRID_RESOLUTION + self.min_bound[2]

            path_marker.points.append(pose.position)

            self.current_path_world.append([pose.position.x, pose.position.y])

        self.pursuit_index = 0
        self.path_publisher.publish(path_marker)

    def timer_cb(self):
        self.listen_to_pointcloud = True

        try:
            # transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            transform = self.tf_buffer.lookup_transform('map', 'gpe', rclpy.time.Time())
            self.robot_foot_location = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            # self.get_logger().warn('Transform not found')
            pass

    def pointcloud2_callback(self, msg):
        if not self.listen_to_pointcloud:
            return
        self.listen_to_pointcloud = False

        try:
            # Convert PointCloud2 to numpy array
            points = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in point_cloud2.read_points(msg, skip_nans=True)])

            if points.size == 0:
                self.get_logger().info('No points received in the point cloud')
                return

            if points.ndim != 2 or points.shape[1] != 3:
                self.get_logger().error(f'Unexpected points shape: {points.shape}')
                return

            # Convert numpy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Create a voxel grid from the point cloud
            voxel_size = self.cell_size  # Use the cell size as the voxel size
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

            # Determine grid dimensions based on voxel grid bounds
            min_bound = voxel_grid.get_min_bound()
            max_bound = voxel_grid.get_max_bound()
            self.min_bound = deepcopy(min_bound)
            
            grid_size_x = int((max_bound[0] - min_bound[0]) / voxel_size) + 1
            grid_size_y = int((max_bound[1] - min_bound[1]) / voxel_size) + 1
            grid_size_z = int((max_bound[2] - min_bound[2]) / voxel_size) + 1

            # Initialize 3D occupancy grid with -1 (unknown)
            occupancy_grid = np.full((grid_size_x, grid_size_y, grid_size_z), -1, dtype=int)

            # Mark only the voxels corresponding to actual points as occupied
            for voxel in voxel_grid.get_voxels():
                grid_index = voxel.grid_index
                x, y, z = grid_index
                
                # Ensure indices are within bounds
                if 0 <= x < grid_size_x and 0 <= y < grid_size_y and 0 <= z < grid_size_z:
                    occupancy_grid[x, y, z] = 100  # Mark the voxel as occupied

            # Store the occupancy grid
            self.occupancy_grid = deepcopy(occupancy_grid)

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')


def main(args=None):
    rclpy.init(args=args)
    pointcloud_to_grid_node = PointCloudToGrid()
    rclpy.spin(pointcloud_to_grid_node)
    pointcloud_to_grid_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
