
def get_center_and_bounds(weighted = False): #22, 23
    if weighted:
        bin_count = {'valence': [27425, 20326, 35916, 46624, 48919, 51225, 56399, 63890, 79380, 75452, 271361,
                                 263633, 169472, 107129, 71913, 63248, 44008, 31391, 23231, 30242],
                     'arousal': [7114, 1499, 1811, 4137, 4832, 6389, 11349, 18920, 29469, 45875, 207891,
                                 306583, 249557, 189182, 136268, 107747, 85861, 64224, 44853, 57623]}
    else:
        bin_count = {'valence': [1] * 20, 'arousal': [1] * 20}

    bin_nums = 20
    bin_centers = {}
    bin_bounds = {}
    for target in bin_count.keys():
        bin_centers[target] = [-1]
        bin_bounds[target] = [-1]
        sample_nums = sum(bin_count[target])
        samples_in_bin = sample_nums / (1.0 * bin_nums)
        previous, current, current_nums = 0, 0, bin_count[target][0]

        for b in range(bin_nums):
            while round(current_nums) < round(samples_in_bin):
                current += 1
                current_nums += bin_count[target][current]

            current_nums -= samples_in_bin
            center = ((previous + current + 1 - current_nums / bin_count[target][current]) / 2.0) / 10.0 - 1.0
            left= previous / 10.0 - 1.0
            right = center * 2 - left
            bin_centers[target].append(center)
            bin_bounds[target].append(left if b != 0 else (-1 + center)/2)
            previous = current + 1 - current_nums / bin_count[target][current]
            #print(left, center, right, previous, current, current_nums)
        bin_centers[target].append(1)
        bin_bounds[target].append((center + 1)/2)
        bin_bounds[target].append(1)

    return bin_centers, bin_bounds


if __name__ == '__main__':
    bin_centers, bin_bounds = get_center_and_bounds(weighted=True)
    print(bin_centers)
    print(bin_bounds)