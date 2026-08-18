cmake_minimum_required(VERSION 3.8)
project(aura_bringup)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
