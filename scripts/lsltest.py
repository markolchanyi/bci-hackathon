"""
Example program to show how to read a
multi-channel time series from LSL.
"""

from pylsl import StreamInlet, resolve_stream


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('source_id', '102801-0077 500')

    for stream in streams:
        print("Stream Name:", stream.name())
        print("Stream Type:", stream.type())
        print("Stream ID:", stream.source_id())   # this should match your X.on Serial Number
        print("Stream Unique Identifier:", stream.uid())
        print("---------------")

    # create a new inlet to read from the stream
    # you'll need to make sure that you're accessing YOUR headset
    inlet = StreamInlet(streams[0])

    while True:
        # get a new sample
        sample, timestamp = inlet.pull_sample()
        print(timestamp, sample)


if __name__ == '__main__':
    main()
