std_msgs/Header header
string task_id
string instruction

# Scene information
sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
sensor_msgs/CameraInfo camera_info

# Optional previous execution context
string previous_error
string previous_feedback
