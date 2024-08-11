# easy_3D_navigation

This repository contains ROS2 nodes for 3D navigation using Octomap, which Team ALeRT used in RRL2024.

For both nodes, first launch https://github.com/RRL-ALeRT/octomap_mapping/blob/ros2/octomap_server/launch/octomap_spot_launch.py

The planned path in both cases can be followed be running
```
ros2 service call /navigation/follow_path std_srvs/srv/SetBool "{data: true}"
```

## plan_3d_2d_path.py
2D occupancy grid is created using the z distance difference between neighbouring topmost cells, which is then used for navigation. Goal can be directly given using 2D initial pose of Rviz2. The issue with this one is, it won't plan under negotiation as the whole thing will be considered as an obstacle in 2D map. This worked perfectly well during RRL2024, without negotiation bars ofcourse.

## plan_3d_path.py
A path is planned through occupied cells. If
1) neighbouring cell is not in a z threshold
2) any cells are found within certain vertical distance above,
3) if, at a certain height, cells are occupied within a certain radius
then that cell is ignored.

Goal can be given using an interactive /goal_marker in Rviz2 and then just publishing to Rviz2's 2D initial pose.
