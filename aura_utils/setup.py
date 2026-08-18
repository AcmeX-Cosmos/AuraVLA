from setuptools import setup, find_packages

package_name = 'aura_utils'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AuraVLA Team',
    maintainer_email='eva@example.com',
    description='AuraVLA utilities and helper functions',
    license='Apache-2.0',
    tests_require=['pytest'],
)
