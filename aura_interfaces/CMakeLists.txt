cmake_minimum_required(VERSION 3.8)
project(aura_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/TaskRequest.msg"
  "msg/TaskPlan.msg"
  "msg/TaskAction.msg"

  # Services
  "srv/EvaluateDoable.srv"
  "srv/GeneratePlan.srv"
  "srv/CheckCompletion.srv"

  # Actions
  "action/ExecuteTask.action"

  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
