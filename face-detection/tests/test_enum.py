from src.metrics.enumerators import MethodAveragePrecision


def test_enum_if():
    method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION
    assert method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION

    if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
        print(1)
    elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
        print(2)
    else:
        print(3)
