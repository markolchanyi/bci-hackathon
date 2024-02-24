import pylsl

# Resolve EEG streams
streams = pylsl.resolve_streams()

# Iterate through streams
print(f"Found {len(streams)} streams")
print("---------------")

for stream in streams:
    print("Stream Name:", stream.name())
    print("Stream Type:", stream.type())
    print("Stream ID:", stream.source_id())   # this should match your X.on Serial Number
    print("Stream Unique Identifier:", stream.uid())
    print("---------------")