from setuptools import setup, find_packages
from glob import glob

package_name = 'aura_planning'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AuraVLA Team',
    maintainer_email='AcmeX@foxmail.com',
    description='AuraVLA planning module for task planning',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planning_node = aura_planning.planning_node:main',
        ],
    },
)
