import json

# считыввание и парсинг файла
all_wall = []
file = open(r"wall_PMI.txt")
for line in file.readlines():
    string = line
    wall = json.loads(string)
    json_decode = json.JSONDecoder()
    parsed_response = json_decode.decode(json.dumps(wall))
    nodes = parsed_response.get('items')
    for node in nodes:
        all_wall.append(node.get("text"))

if __name__ == "__main__":
    print(all_wall[20])
    # подсчет слов
    wordcount = {}
    for wall in all_wall:
        for word in wall.split():
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    try:
        for wc in wordcount:
            print(wc, wordcount[wc])
    except Exception:
        k = 1