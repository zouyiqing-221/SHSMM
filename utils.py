import math
import numpy as np
EARTH_MEAN_RADIUS_METER = 6371008.7714

class SPoint:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    def __str__(self):
        return '({},{})'.format(self.lat, self.lng)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.lat == other.lat and self.lng == other.lng

    def __ne__(self, other):
        return not self == other


class MBR:
    def __init__(self, min_lat, min_lng, max_lat, max_lng):
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng

    def contains(self, lat, lng):
        # return self.min_lat <= lat <= self.max_lat and self.min_lng <= lng <= self.max_lng
        # to be consist with grid index
        return self.min_lat <= lat < self.max_lat and self.min_lng <= lng < self.max_lng

    def center(self):
        return (self.min_lat + self.max_lat) / 2.0, (self.min_lng + self.max_lng) / 2.0

    def get_h(self):
        return distance(SPoint(self.min_lat, self.min_lng), SPoint(self.max_lat, self.min_lng))

    def get_w(self):
        return distance(SPoint(self.min_lat, self.min_lng), SPoint(self.min_lat, self.max_lng))

    def __str__(self):
        h = self.get_h()
        w = self.get_w()
        return '{}x{}m2'.format(h, w)


class Grid:
    """
    index order
    30 31 32 33 34...
    20 21 22 23 24...
    10 11 12 13 14...
    00 01 02 03 04...
    """
    def __init__(self, mbr, row_num, col_num):
        self.mbr = mbr
        self.row_num = row_num
        self.col_num = col_num
        self.lat_interval = (mbr.max_lat - mbr.min_lat) / float(row_num)
        self.lng_interval = (mbr.max_lng - mbr.min_lng) / float(col_num)

    def get_row_idx(self, lat):
        row_idx = int((lat - self.mbr.min_lat) // self.lat_interval)
        if row_idx >= self.row_num or row_idx < 0:
            raise IndexError("lat is out of mbr")
        return row_idx

    def get_col_idx(self, lng):
        col_idx = int((lng - self.mbr.min_lng) // self.lng_interval)
        if col_idx >= self.col_num or col_idx < 0:
            raise IndexError("lng is out of mbr")
        return col_idx

    def safe_matrix_to_idx(self, lat, lng):
        try:
            return self.get_matrix_idx(lat, lng)
        except IndexError:
            return np.nan, np.nan

    def get_idx(self, lat, lng):
        return self.get_row_idx(lat), self.get_col_idx(lng)

    def get_matrix_idx(self, lat, lng):
        return self.row_num - 1 - self.get_row_idx(lat), self.get_col_idx(lng)


def distance(a, b):
    return haversine_distance(a, b)


def haversine_distance(a, b):
    if same_coords(a, b):
        return 0.0
    delta_lat = math.radians(b.lat - a.lat)
    delta_lng = math.radians(b.lng - a.lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a.lat)) * math.cos(
        math.radians(b.lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


def same_coords(a, b):
    if a.lat == b.lat and a.lng == b.lng:
        return True
    else:
        return False