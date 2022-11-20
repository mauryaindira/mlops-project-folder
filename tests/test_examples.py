# examples of unit test cases
import math
import numpy as np

def sum(x, y):
    return x+y

def test_sum():
    x = 10
    y = 20
    z = sum(x,y)
    expected = 30
    assert z == expected

def test_equal():
    assert 1==1


def test_sqrt():
    num = 36
    assert math.sqrt(num) == 6, "Sqrt failed"


def test_square():
    num = 5
    assert num**2 == 25, "Square failed"


def test_equality():
    assert 20 == 20, "not equal"



