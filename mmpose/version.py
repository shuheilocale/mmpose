# Copyright (c) Open-MMLab. All rights reserved.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
__version__ = '1.0.0'
=======
__version__ = '0.27.0'
>>>>>>> 94f1dbcb (Bump version v0.27.0 (#1414))
=======
__version__ = '0.28.0'
>>>>>>> 5123a2a7 (Bump version 0.28.0 (#1468))
=======
__version__ = '0.28.1'
>>>>>>> efdfa1c8 (bump version v0.28.1)
short_version = __version__


def parse_version_info(version_str):
    """Parse a version string into a tuple.

    Args:
        version_str (str): The version string.
    Returns:
        tuple[int | str]: The version info, e.g., "1.3.0" is parsed into
            (1, 3, 0), and "2.0.0rc1" is parsed into (2, 0, 0, 'rc1').
    """
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
        elif x.find('b') != -1:
            patch_version = x.split('b')
            version_info.append(int(patch_version[0]))
            version_info.append(f'b{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
