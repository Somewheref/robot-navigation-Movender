#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf.transformations

class Visualizer:
    def __init__(self):
        # Initialize publishers
        self.marker_pub = rospy.Publisher('scan_markers', MarkerArray, queue_size=1)
        self.grid_marker_pub = rospy.Publisher('grid_markers', MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher('path_marker', Marker, queue_size=1)
        self.sphere_pub = rospy.Publisher('path_spheres', Marker, queue_size=1)
        self.pose_marker_pub = rospy.Publisher('robot_pose_marker', Marker, queue_size=1)
        # Add new publisher for emergency stop visualization
        self.emergency_stop_pub = rospy.Publisher('emergency_stop_zone', Marker, queue_size=1)
        # Add new publisher for clear zone visualization
        self.clear_zone_pub = rospy.Publisher('clear_zone', Marker, queue_size=1)

    def visualize_points(self, scan_points, frame_id="base_link"):
        """Visualize laser scan points."""
        markers = MarkerArray()
        
        for i, point in enumerate(scan_points):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point['x']
            marker.pose.position.y = point['y']
            marker.pose.position.z = 0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.a = 1.0
            
            markers.markers.append(marker)
        
        self.marker_pub.publish(markers)

    def visualize_occupancy_grid(self, grid_manager, frame_id="base_link"):
        """Visualize the occupancy grid."""
        markers = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)
        
        # Create a marker for each occupied cell
        marker_id = 1
        for i in range(grid_manager.grid_size):
            for j in range(grid_manager.grid_size):
                if grid_manager.occupancy_grid[i][j]:
                    cell_marker = Marker()
                    cell_marker.header.frame_id = frame_id
                    cell_marker.header.stamp = rospy.Time.now()
                    cell_marker.id = marker_id
                    marker_id += 1
                    cell_marker.type = Marker.CUBE
                    cell_marker.action = Marker.ADD
                    
                    # Get world coordinates
                    world_x, world_y = grid_manager.grid_to_world(i, j)
                    cell_marker.pose.position.x = world_x
                    cell_marker.pose.position.y = world_y
                    cell_marker.pose.position.z = 0
                    
                    cell_marker.scale.x = grid_manager.GRID_RESOLUTION * 0.9
                    cell_marker.scale.y = grid_manager.GRID_RESOLUTION * 0.9
                    cell_marker.scale.z = 0.1
                    
                    cell_marker.color.r = 1.0
                    cell_marker.color.g = 0.0
                    cell_marker.color.b = 0.0
                    cell_marker.color.a = 0.5
                    
                    # Set orientation
                    cell_marker.pose.orientation.w = 1.0
                    
                    markers.markers.append(cell_marker)
        
        self.grid_marker_pub.publish(markers)

    def visualize_robot_pose(self, grid_manager, frame_id="base_link"):
        """Visualize the robot's pose as a fixed arrow marker at the center."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Fixed position at the center
        center_x, center_y = grid_manager.grid_to_world(grid_manager.grid_size // 2, grid_manager.grid_size // 2)
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = 0.1
        
        # Fixed orientation pointing forward
        quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.2  # Arrow width
        marker.scale.z = 0.2  # Arrow height
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.pose_marker_pub.publish(marker)

    def visualize_path(self, path, grid_manager):
        """Visualize the planned path."""
        if not path or len(path) < 2:
            #rospy.loginfo("No path or too few points to visualize")
            return

        # Line strip marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Line width
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for point in path:
            world_x, world_y = grid_manager.grid_to_world(point[0], point[1])
            p = Point()
            p.x = world_x
            p.y = world_y
            p.z = 0
            marker.points.append(p)

        self.path_pub.publish(marker)

        # Sphere marker for path points
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "base_link"
        sphere_marker.header.stamp = rospy.Time.now()
        sphere_marker.id = 1
        sphere_marker.type = Marker.SPHERE_LIST
        sphere_marker.action = Marker.ADD
        sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.1
        sphere_marker.color.b = 1.0
        sphere_marker.color.a = 1.0
        
        for point in path:
            world_x, world_y = grid_manager.grid_to_world(point[0], point[1])
            p = Point()
            p.x = world_x
            p.y = world_y
            p.z = 0
            sphere_marker.points.append(p)
        
        self.sphere_pub.publish(sphere_marker)

    def visualize_emergency_stop_zone(self, emergency_stop_distance, frame_id="base_link"):
        """Visualize the emergency stop zone as a transparent circle."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position at robot's center
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        
        # Set orientation (flat cylinder)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set scale (diameter = 2 * emergency_stop_distance)
        marker.scale.x = emergency_stop_distance * 2
        marker.scale.y = emergency_stop_distance * 2
        marker.scale.z = 0.01  # Very thin cylinder
        
        # Set color (red with 20% opacity)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.2
        
        self.emergency_stop_pub.publish(marker)

    def visualize_clear_zone(self, grid_manager, frame_id="base_link"):
        """Visualize the clear zone as a shifted square."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Calculate the clear zone corners in world coordinates
        size = grid_manager.clear_zone_size * grid_manager.GRID_RESOLUTION
        offset = grid_manager.clear_zone_offset * grid_manager.GRID_RESOLUTION
        
        # Define the corners of the square (shifted by offset on x-axis)
        points = [
            Point(offset + size, size, 0),  # Top right
            Point(offset + size, -size, 0),  # Bottom right
            Point(offset - size, -size, 0),  # Bottom left
            Point(offset - size, size, 0),  # Top left
            Point(offset + size, size, 0),  # Back to start to close the square
        ]
        
        marker.points = points
        
        # Set the line properties
        marker.scale.x = 0.02  # Line width
        marker.color.g = 1.0   # Green color
        marker.color.a = 0.8   # 80% opacity
        
        # Set orientation
        marker.pose.orientation.w = 1.0
        
        # Add lifetime to marker
        marker.lifetime = rospy.Duration(0.5)
        
        self.clear_zone_pub.publish(marker) 