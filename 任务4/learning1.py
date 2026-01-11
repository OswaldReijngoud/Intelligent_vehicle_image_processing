from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# 使用方法
current_color = Color.RED

if current_color == Color.RED:
    print("停止")

print(Color.RED.name)   # 输出: RED (名称)
print(Color.RED.value)  # 输出: 1 (对应的值)