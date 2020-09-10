import csv

Parameter = ['K', 'p', 'Nk', 'Loss_f', 'Lambda', 'Alpha', 'q']
Data = ['Loss_train', 'Loss_test', 'Ac_train', 'Ac_test', 'non_zero', 'i']
rep = 100
with open('grand_sum.csv', 'w', newline='') as g_file:
    writer = csv.DictWriter(g_file, Parameter + Data)
    with open('./results/env=Sim/sum.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        sorted_reader = sorted(reader, key=lambda row: list(map(lambda x: row[x], Parameter + ['id'])), reverse=False)
        writer.writerow({x: x for x in Parameter + Data})
        fieldnames = Parameter + Data
        i = 0
        local_data = {x: 0.0 for x in Data}
        for row in sorted_reader:
            if i == rep:
                sum_data = {x: local_data[x] / rep for x in Data}
                local_data = {x: 0.0 for x in Data}
                i = 0
                writer.writerow({**row_info, **sum_data})
                print(row["id"])
            for x in Data:
                local_data[x] += float(row[x])
            if int(i) != int(row['id']):
                pass
            row_info = {x: row[x] for x in Parameter}
            i += 1
