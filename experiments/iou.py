def calculateIoUs(result_path):
    iouDict = {} #key is the city, value is a list [list of iou's, iou sum]
    with open(result_path) as f:
        for line in f.readlines():
            if not line[0].isdigit():
                split = line.rstrip("\n").split()
                end = len(split[0]) - 1
                intersect = split[1][1:len(split[1]) - 1]
                union = split[2][0:len(split[2]) - 1]
                iou = float(intersect) / float(union)
                while(split[0][end].isdigit()):
                    end -= 1
                if(split[0][0:end + 1] in iouDict):
                    iouDict[split[0][0:end + 1]][0].append(iou)
                    iouDict[split[0][0:end + 1]][1] += iou
                else:
                    iouDict[split[0][0:end + 1]] = [[iou], iou]
        for key, val in iouDict.items():
            print(key, " mean IoU: ", val[1] / len(val[0]))

def weightedIoUs(result_path):
    iouDict = {} #key is the city, value is a list [list of tuples (intersection, union), intersection sum, union sum]
    with open(result_path) as f:
        for line in f.readlines():
            if not line[0].isdigit():
                split = line.rstrip("\n").split()
                end = len(split[0]) - 1
                intersect = float(split[1][1:len(split[1]) - 1])
                union = float(split[2][0:len(split[2]) - 1])
                while(split[0][end].isdigit()):
                    end -= 1
                if(split[0][0:end + 1] in iouDict):
                    iouDict[split[0][0:end + 1]][0].append((intersect,union))
                    iouDict[split[0][0:end + 1]][1] += intersect
                    iouDict[split[0][0:end + 1]][2] += union
                else:
                    iouDict[split[0][0:end + 1]] = [[(intersect,union)], intersect, union]
    for key, val in iouDict.items():
        print(key, " mean IoU: ", val[1]/val[2])

if __name__ == '__main__':
    #calculateIoUs("/Users/paulrhee/Desktop/example.txt")
    weightedIoUs('/Users/Varun/Documents/CityEngine/Synthetic_city_generation/experiments/results/rV_result_2.txt')
    #calculateIoUs("/Users/paulrhee/Desktop/Data+/result.txt")
