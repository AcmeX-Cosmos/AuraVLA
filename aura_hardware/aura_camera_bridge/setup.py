from setuptools import setup

setup(
    name='aura_camera_bridge',
    version='1.0.0',
    packages=[],
    py_modules=['camera_bridge_node', 'camera_bridge'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/aura_camera_bridge']),
        ('share/aura_camera_bridge', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AuraVLA Team',
    maintainer_email='AcmeX@foxmail.com',
    description='AuraVLA camera bridge',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_bridge_node = camera_bridge_node:main',
        ],
    },
)
