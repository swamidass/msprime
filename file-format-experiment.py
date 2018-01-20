"""
Experimental work for a simple key->numeric array store.
"""
import numpy as np
import struct


# This is a rubbish magic number. Just generate something randomly for the
# real thing.
MAGIC = bytes([0, 1, 2, 3, 4, 5, 6, 7])
VERSION_MAJOR = 0
VERSION_MINOR = 1


np_dtype_to_type_map = {
    # Obviously support more of these...
    "int8": 1,
    "uint32": 2,
    "float64": 3,
}

type_to_np_dtype_map = {t: dtype for dtype, t in np_dtype_to_type_map.items()}


class ItemDescriptor(object):
    """
    The information required to recover a single key-value pair from the
    file.
    """
    size = 64

    def __init__(self, type_, key_start, key_len, array_start, array_len):
        self.type = type_
        self.key_start = key_start
        self.key_len = key_len
        self.array_start = array_start
        self.array_len = array_len
        self.key = None
        self.array = None

    def __str__(self):
        return "type={};key_start={};key_len={};array_start={};array_len={}".format(
                self.type, self.key_start, self.key_len, self.array_start,
                self.array_len)

    def pack(self):
        descriptor = bytearray(64)
        descriptor[0:4] = struct.pack("<I", self.type)
        descriptor[4:12] = struct.pack("<Q", self.key_start)
        descriptor[12:20] = struct.pack("<Q", self.key_len)
        descriptor[20:28] = struct.pack("<Q", self.array_start)
        descriptor[28:36] = struct.pack("<Q", self.array_len)
        # bytes 36:64 are reserved.
        return descriptor

    @classmethod
    def unpack(cls, descriptor):
        type_ = struct.unpack("<I", descriptor[0:4])[0]
        key_start = struct.unpack("<Q", descriptor[4:12])[0]
        key_len = struct.unpack("<Q", descriptor[12:20])[0]
        array_start = struct.unpack("<Q", descriptor[20:28])[0]
        array_len = struct.unpack("<Q", descriptor[28:36])[0]
        return cls(type_, key_start, key_len, array_start, array_len)


def write_arrays(filename, arrays):
    """
    Writes the arrays in the specified mapping to the key-array-store file.
    """
    with open(filename, "wb") as f:
        num_items = len(arrays)
        header_size = 64
        header = bytearray(header_size)
        header[0:8] = MAGIC
        header[8:12] = struct.pack("<I", VERSION_MAJOR)
        header[12:16] = struct.pack("<I", VERSION_MINOR)
        header[16:20] = struct.pack("<I", num_items)
        # The rest of the header is reserved.
        f.write(header)

        # We store the keys in sorted order.
        sorted_keys = sorted(arrays.keys())
        total_keylen = sum(len(key) for key in sorted_keys)
        descriptor_block_size = num_items * ItemDescriptor.size
        key_start = header_size + descriptor_block_size
        array_start = key_start + total_keylen
        descriptors = []
        for key, array in arrays.items():
            assert len(array.shape) == 1  # Only 1D arrays supported.
            descriptor = ItemDescriptor(
                np_dtype_to_type_map[array.dtype.name],
                key_start, len(key), array_start, array.nbytes)
            descriptor.key = key
            descriptor.array = array
            key_start += len(key)
            array_start += array.nbytes
            descriptors.append(descriptor)

        assert f.tell() == header_size
        # Now write the descriptors.
        for descriptor in descriptors:
            f.write(descriptor.pack())

        # Write the keys
        for descriptor in descriptors:
            assert f.tell() == descriptor.key_start
            f.write(descriptor.key.encode())

        # Write the arrays
        for descriptor in descriptors:
            assert f.tell() == descriptor.array_start
            f.write(descriptor.array.data)


def read_arrays(filename):
    """
    Reads arrays from the specified file and returns the resulting mapping.
    """
    print("READ")
    with open(filename, "rb") as f:
        header_size = 64
        header = f.read(header_size)
        if header[0:8] != MAGIC:
            raise ValueError("Incorrect file format")
        version_major = struct.unpack("<I", header[8:12])[0]
        version_minor = struct.unpack("<I", header[12:16])[0]
        print("version_major = ", version_major)
        if version_major != VERSION_MAJOR:
            raise ValueError("Incompatible major version")
        num_items = struct.unpack("<I", header[16:20])[0]

        descriptor_block_size = num_items * ItemDescriptor.size
        descriptor_block = f.read(descriptor_block_size)

        offset = 0
        descriptors = []
        for _ in range(num_items):
            descriptor = ItemDescriptor.unpack(
                descriptor_block[offset: offset + ItemDescriptor.size])
            descriptors.append(descriptor)
            offset += ItemDescriptor.size

        keys = []
        for descriptor in descriptors:
            assert f.tell() == descriptor.key_start
            keys.append(f.read(descriptor.key_len).decode())

        arrays = []
        for descriptor in descriptors:
            assert f.tell() == descriptor.array_start
            dtype = type_to_np_dtype_map[descriptor.type]
            data = f.read(descriptor.array_len)
            array = np.frombuffer(data, dtype=dtype)
            arrays.append(array)
        return dict(zip(keys, arrays))





def main():
    arrays = {
        "one/two": np.arange(10, dtype=np.int8),
        "three/four/five": np.zeros(20, dtype=np.uint32) + 1,
        "six": np.zeros(5, dtype=np.float64) + 2,
    }

    # Key-array-store
    filename = "tmp.kas"
    write_arrays(filename, arrays)

    other = read_arrays(filename)
    assert list(other.keys()) == list(arrays.keys())
    for k in other.keys():
        assert np.array_equal(arrays[k], other[k])




if __name__ == "__main__":
    main()
