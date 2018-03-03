"""
Experimental work for a simple key->numeric array store.
"""
import struct
import os.path
import argparse

import numpy as np
import humanize
import msprime


# This is a rubbish magic number. Just generate something randomly for the
# real thing.
MAGIC = bytes([0, 1, 2, 3, 4, 5, 6, 7])
VERSION_MAJOR = 0
VERSION_MINOR = 1


np_dtype_to_type_map = {
    # Obviously support more of these...
    "int8": 1,
    "uint32": 2,
    "int32": 3,
    "float64": 4,
}

type_to_np_dtype_map = {t: dtype for dtype, t in np_dtype_to_type_map.items()}


class ItemDescriptor(object):
    """
    The information required to recover a single key-value pair from the
    file. Each descriptor is a block of 64 bytes, which stores:

    - The numeric type of the array (similar to numpy dtype)
    - The start offset of the key
    - The length of the key
    - The start offset of the array
    - The length of the array

    File offsets are stored as 8 byte unsigned little endian integers.
    The remaining space in the descriptor is reserved for later use.
    For example, we may wish to add an 'encoding' field in the future,
    which allows for things like simple run-length encoding and so on.
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
        # It's a bit ridiclous having 4 bytes for the type really.
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
        # TODO Change this so that we write all the keys in one block and
        # all the arrays afterwards. This is to allow us to efficiently read
        # in the names of the arrays without seeking all over the file.
        sorted_keys = sorted(arrays.keys())
        descriptor_block_size = num_items * ItemDescriptor.size
        offset = header_size + descriptor_block_size
        descriptors = []
        for key in sorted_keys:
            array = arrays[key]
            assert len(array.shape) == 1  # Only 1D arrays supported.
            key_start = offset
            array_start = key_start + len(key) # TODO Add padding to 8-align
            descriptor = ItemDescriptor(
                np_dtype_to_type_map[array.dtype.name],
                key_start, len(key), array_start, array.nbytes)
            descriptor.key = key
            descriptor.array = array
            descriptors.append(descriptor)
            offset = array_start + array.nbytes  # TODO Add padding to 8-align

        assert f.tell() == header_size
        # Now write the descriptors.
        for descriptor in descriptors:
            f.write(descriptor.pack())

        # Write the keys and arrays
        for descriptor in descriptors:
            assert f.tell() == descriptor.key_start
            f.write(descriptor.key.encode())
            assert f.tell() == descriptor.array_start
            f.write(descriptor.array.data)


def read_header(f):
    """
    Reads the header from the specified stream and returns the list
    of descriptors.
    """
    header_size = 64
    header = f.read(header_size)
    if header[0:8] != MAGIC:
        raise ValueError("Incorrect file format")
    version_major = struct.unpack("<I", header[8:12])[0]
    version_minor = struct.unpack("<I", header[12:16])[0]
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
        f.seek(descriptor.key_start)
        descriptor.key = f.read(descriptor.key_len).decode()
    return descriptors


def read_arrays(filename):
    """
    Reads arrays from the specified file and returns the resulting mapping.
    """
    with open(filename, "rb") as f:
        descriptors = read_header(f)

        items = {}
        for descriptor in descriptors:
            # TODO this should skip reading the keys, as this should be done as
            # part of the header reading process.
            # assert f.tell() == descriptor.key_start
            # descriptor.key = f.read(descriptor.key_len).decode()
            # assert f.tell() ==
            f.seek(descriptor.array_start)
            dtype = type_to_np_dtype_map[descriptor.type]
            data = f.read(descriptor.array_len)
            descriptor.array = np.frombuffer(data, dtype=dtype)
            items[descriptor.key] = descriptor.array
        return items


def print_summary(filename):
    """
    Prints out a summary of the specified kas file.
    """
    # Definitely want a class so that we can do this sort of stuff
    # with more control.
    with open(filename, "rb") as f:
        descriptors = read_header(f)

    print("Summary for", filename)
    print("Total size = ", humanize.naturalsize(os.path.getsize(filename), binary=True))
    print("Num arrays = ", len(descriptors))
    key_width = max(len(descriptor.key) for descriptor in descriptors)
    for descriptor in descriptors:
        print(
            "{:<{}}".format(descriptor.key, key_width), #str(descriptor),
            humanize.naturalsize(descriptor.array_len, binary=True))



def hdf5_to_kas(hdf5_file, kas_file):

    ts = msprime.load(hdf5_file)
    tables = ts.tables

    arrays = {
        "sequence_length": np.array([ts.sequence_length]),

        "nodes/flags": tables.nodes.flags,
        "nodes/time": tables.nodes.time,
        "nodes/population": tables.nodes.population,
        "nodes/metadata": tables.nodes.metadata,
        "nodes/metadata_offset": tables.nodes.metadata_offset,

        "edges/left": tables.edges.left,
        "edges/right": tables.edges.right,
        "edges/parent": tables.edges.parent,
        "edges/child": tables.edges.child,

        "migrations/left": tables.migrations.left,
        "migrations/right": tables.migrations.right,
        "migrations/time": tables.migrations.time,
        "migrations/node": tables.migrations.node,
        "migrations/source": tables.migrations.source,
        "migrations/dest": tables.migrations.dest,

        "sites/position": tables.sites.position,
        "sites/ancestral_state": tables.sites.ancestral_state,
        "sites/ancestral_state_offset": tables.sites.ancestral_state_offset,
        "sites/metadata": tables.sites.metadata,
        "sites/metadata_offset": tables.sites.metadata_offset,

        "mutations/site": tables.mutations.site,
        "mutations/node": tables.mutations.node,
        "mutations/derived_state": tables.mutations.derived_state,
        "mutations/derived_state_offset": tables.mutations.derived_state_offset,
        "mutations/metadata": tables.mutations.metadata,
        "mutations/metadata_offset": tables.mutations.metadata_offset,

        "provenances/timestamp": tables.provenances.timestamp,
        "provenances/timestamp_offset": tables.provenances.timestamp_offset,
        "provenances/record": tables.provenances.record,
        "provenances/record_offset": tables.provenances.record_offset,
    }

    write_arrays(kas_file, arrays)

def load_tree_sequence(filename):
    """
    Loads a tree sequence from the specified kas file.
    """
    items = read_arrays(filename)
    sequence_length = items["sequence_length"][0]

    nodes = msprime.NodeTable()
    nodes.set_columns(
        flags=items["nodes/flags"],
        time=items["nodes/time"],
        population=items["nodes/population"],
        metadata=items["nodes/metadata"],
        metadata_offset=items["nodes/metadata_offset"])

    edges = msprime.EdgeTable()
    edges.set_columns(
        left=items["edges/left"],
        right=items["edges/right"],
        parent=items["edges/parent"],
        child=items["edges/child"])

    migrations = msprime.MigrationTable()
    migrations.set_columns(
        left=items["migrations/left"],
        right=items["migrations/right"],
        node=items["migrations/node"],
        source=items["migrations/source"],
        dest=items["migrations/dest"],
        time=items["migrations/time"])

    sites = msprime.SiteTable()
    sites.set_columns(
        position=items["sites/position"],
        ancestral_state=items["sites/ancestral_state"],
        ancestral_state_offset=items["sites/ancestral_state_offset"],
        metadata=items["sites/metadata"],
        metadata_offset=items["sites/metadata_offset"])

    mutations = msprime.MutationTable()
    mutations.set_columns(
        site=items["mutations/site"],
        node=items["mutations/node"],
        derived_state=items["mutations/derived_state"],
        derived_state_offset=items["mutations/derived_state_offset"],
        metadata=items["mutations/metadata"],
        metadata_offset=items["mutations/metadata_offset"])

    provenances = msprime.ProvenanceTable()
    provenances.set_columns(
        timestamp=items["provenances/timestamp"],
        timestamp_offset=items["provenances/timestamp_offset"],
        record=items["provenances/record"],
        record_offset=items["provenances/record_offset"])

    return msprime.load_tables(
        nodes=nodes, edges=edges, migrations=migrations, sites=sites,
        mutations=mutations, provenances=provenances, sequence_length=sequence_length)


def check_ts(ts1, ts2):
    assert ts1.sequence_length == ts2.sequence_length
    assert ts1.tables.nodes == ts2.tables.nodes
    assert ts1.tables.edges == ts2.tables.edges
    assert ts1.tables.migrations == ts2.tables.migrations
    assert ts1.tables.sites == ts2.tables.sites
    assert ts1.tables.mutations == ts2.tables.mutations
    assert ts1.tables.provenances == ts2.tables.provenances

def run_convert(args):

    hdf5_to_kas(args.input, args.output)

    ts = load_tree_sequence(args.output)
    check_ts(ts, msprime.load(args.input))


def run_ls(args):
    print_summary(args.input)

def main():
    top_parser = argparse.ArgumentParser(
        description="Toolkit for key-array-store files.")
    top_parser.add_argument(
        "-V", "--version", action='version',
        version='%(prog)s {}'.format("0.0.1"))
    # This is required to get uniform behaviour in Python2 and Python3
    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser = subparsers.add_parser(
        "convert",
        help="Convert msprime HDF5 file to kas.")
    parser.add_argument("input",
        help="Input msprime HDF5 file.")
    parser.add_argument("output",
        help="Output KAS file.")
    parser.set_defaults(runner=run_convert)

    parser = subparsers.add_parser(
        "ls",
        help="Lists the contents of the KAS file.")
    parser.add_argument("input",
        help="Input KAS file.")
    parser.set_defaults(runner=run_ls)

    args = top_parser.parse_args()
    args.runner(args)


if __name__ == "__main__":
    main()
