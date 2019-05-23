"""
"""

import phoebe

def test_all():
    print("phoebe.Bundle() ...")
    b = phoebe.Bundle()

    for force_build in [True, False]:
        print("phoebe.default_star(force_build={}) ...".format(force_build))
        b = phoebe.default_star(force_build=force_build)
        print("phoebe.default_binary(force_build={}) ...".format(force_build))
        b = phoebe.default_binary(force_build=force_build)
        print("phoebe.default_binary(contact_envelope=True, force_build={}) ...".format(force_build))
        b = phoebe.default_binary(contact_envelope=True, force_build=force_build)
        if force_build:
            print("phoebe.default_triple(hierarchy='21', force_build={}) ...".format(force_build))
            b = phoebe.default_triple(hierarchy='21', force_build=force_build)
            print("phoebe.default_triple(hierarchy='12', force_build={}) ...".format(force_build))
            b = phoebe.default_triple(hierarchy='12', force_build=force_build)
            print("phoebe.default_triple(hierarchy='12', contact_envelope=True, force_build={}) ...".format(force_build))
            b = phoebe.default_triple(hierarchy='12', contact_envelope=True, force_build=force_build)


if __name__ == '__main__':
    logger = phoebe.logger(clevel='WARNING')

    test_all()
