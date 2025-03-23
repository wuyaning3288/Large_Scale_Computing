lines_rdd = sc.textFile("pulsar.dat")
parsed_rdd = lines_rdd.map(lambda line: [float(x) for x in line.split()])

position_error = 0.2  # degrees
frequency_error = 0.2  # MHz

# Create a function to check if two signals are the same
def are_signals_similar(signal_a, signal_b):
    ascension_a, declination_a, _, frequency_a = signal_a
    ascension_b, declination_b, _, frequency_b = signal_b
    return (
        abs(ascension_a - ascension_b) < position_error and
        abs(declination_a - declination_b) < position_error and
        abs(frequency_a - frequency_b) < frequency_error
    )

# Generate Cartesian product of parsed_rdd with itself
cartesian_rdd = parsed_rdd.cartesian(parsed_rdd)

# Filter signal pairs that aligns
similar_pairs_rdd = cartesian_rdd.filter(lambda pair: are_signals_similar(pair[0], pair[1]) )
#Set key
grouped_signals_rdd = similar_pairs_rdd.map(lambda pair: (tuple([pair[0][0], pair[0][1], pair[0][3]]), pair[1][2]))
blip_counts_rdd = grouped_signals_rdd.groupByKey().mapValues(list).mapValues(len)
max_group = blip_counts_rdd.max(key=lambda x: x[1])
max_key = max_group[0]

# Find the time period
time_values = grouped_signals_rdd.filter(lambda x: x[0] == max_key).map(lambda x: x[1])
all_times = time_values.collect()
time_difference = max(all_times) - min(all_times)

print(f"There are {max_group[1]} blips at  {max_group[0]} .")
print(f"All time values in the max group: {all_times}")
print(f"Difference between max and min time values: {time_difference}")