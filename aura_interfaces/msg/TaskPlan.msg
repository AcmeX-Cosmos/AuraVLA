std_msgs/Header header
string task_id
string instruction

# Plan metadata
string rationale
float32 confidence
string[] constraints

# Actions
TaskAction[] actions
