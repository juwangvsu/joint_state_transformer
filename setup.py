from setuptools import find_packages, setup

import os, glob

package_name = 'joint_state_transformer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "launch"),
            glob.glob(os.path.join("launch", "launch.*.py")),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yong Tang',
    maintainer_email='yong.tang.github@outlook.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'transformer = joint_state_transformer.transformer:main',
        ],
    },
)
