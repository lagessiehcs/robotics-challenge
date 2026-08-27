FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-slam-toolbox \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-rviz2 \
    ros-humble-teleop-twist-keyboard \
    python3-colcon-common-extensions \
    python3-pip \
    python3-numpy \
    python3-yaml \
    python3-pil \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /challenge/ws

# docker-compose.yml bind-mounts ws/ at run time; build with `colcon build` inside the running container.

ENV ROS_DOMAIN_ID=42

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /challenge/ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
