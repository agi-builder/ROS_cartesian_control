# ROS_projects
This is the repository for robotics course projects.

For more infomantion please go to https://xiangzhuo-ding.github.io/ROS_Projects.html

## Requirements:
1. Environment: Ubuntu 16.04
2. ROS Kinetic
3. xterm

## Set up the workspace:
1. Go to each project.
2. Run ```catkin_make```

## Let's run it!
1. Run ```source devel/setup.bash``` for each terminal
2. Run the simulator 
    ```roslaunch cartesian_control kuka_lwr.launch```
    or
    ```roslaunch cartesian_control ur5.launch```
3. Now you can control the robot by running
    ```rosrun run ccik.py```
