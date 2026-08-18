from setuptools import setup

setup(
    name='aura_isaac_bridge',
    version='1.0.0',
    packages=[],
    py_modules=['isaac_bridge_node'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AuraVLA Team',
    maintainer_email='AcmeX@foxmail.com',
    description='AuraVLA Isaac Sim bridge',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'isaac_bridge_node = isaac_bridge_node:main',
        ],
    },
)
