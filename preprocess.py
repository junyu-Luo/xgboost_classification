import pandas as pd
import numpy as np


def add_data(dataset, dataset_id_list, add_columns, seq_id='BLINDID', add_dir='./data/outcome2.xlsx', sheet_name='Sheet1'):
    add_DF = pd.read_excel(add_dir, sheetname=sheet_name)
    DF_len = len(add_DF)
    try:
        add_DF[add_columns]
    except:
        add_DF[add_columns] = [np.nan] * len(add_DF)

    id_list = []
    add_list = []
    for i in range(DF_len):
        id_list.append(add_DF[seq_id][i])
        add_list.append(add_DF[add_columns][i])
    print('转换列表完毕')  # 读取列表字典比DF快

    for i in range(DF_len):
        print(add_columns, "数据正在添加", round(i / DF_len, 2), i)
        for j in range(len(dataset)):
            if dataset_id_list[j] == id_list[i]:
                try:
                    if np.isnan(add_list[i]) == False:
                        dataset[add_columns][j] = add_list[i]
                except:
                    if type(add_list[i]) != float:
                        dataset[add_columns][j] = add_list[i]

    return dataset


def get_median(dataset, column_name):
    column_exist_nan = []
    for i in range(len(dataset)):
        column_exist_nan.append(dataset[column_name][i])

    column_no_nan = []
    for i in range(len(column_exist_nan)):
        try:
            if np.isnan(column_exist_nan[i]) == False:
                column_no_nan.append(column_exist_nan[i])
        except:
            if type(column_exist_nan[i]) != float:
                column_no_nan.append(column_exist_nan[i])

    return len(column_no_nan), np.median(np.array(column_no_nan))


def full_nan(dataset, columns, dataset_id_list):
    id_len = len(dataset_id_list)
    for column in columns:
        col_len, median = get_median(dataset, column)
        if col_len != id_len:
            if col_len / id_len >= 0.9:
                for i in range(id_len):
                    print(column, "数据正在添加", round(i / id_len, 2), i)

                    try:
                        if np.isnan(dataset[column][i]):
                            dataset[column][i] = median

                    except:
                        if type(dataset[column][i]) == float:
                            dataset[column][i] = median

    return dataset


if __name__ == '__main__':
    # 第一行第一列不要为空
    dataset = pd.read_excel('./data/dataset.xlsx')

    dataset_id_list = []
    dataset_len = len(dataset)
    for i in range(dataset_len):
        dataset_id_list.append(dataset['BLINDID'][i])  # 若原始的和添加的不同需要修改 seq_id

    # 填充空值
    # columns = [column for column in dataset]
    # for delete in ['BLINDID', 'PEX_TOT0101', 'PEX_DRUGTOT0101', 'PEX_SEVERETOT0101', 'PEX_TOT_365','PEX_SEVERE_365', 'RDF25', 'RDS20', 'RDS21', 'PEX_DRUG_365']:
    #     columns.remove(delete)
    # dataset = full_nan(dataset=dataset, columns=columns, dataset_id_list=dataset_id_list)
    # dataset.to_excel('out.xlsx')

    # 添加数据
    dataset = add_data(dataset=dataset, dataset_id_list=dataset_id_list, add_columns='PEX_TOT0102', sheet_name='Sheet1')
    # dataset = add_data(dataset=dataset, dataset_id_list=dataset_id_list, add_columns='PEX_DRUGTOT0101',
    #                    sheet_name='Sheet1')
    # dataset = add_data(dataset=dataset, dataset_id_list=dataset_id_list, add_columns='PEX_SEVERETOT0101',
    #                    sheet_name='Sheet1')
    # dataset = add_data(dataset=dataset, dataset_id_list=dataset_id_list, add_columns='PEX_TOT_365', sheet_name='Sheet1')
    # dataset = add_data(dataset=dataset, dataset_id_list=dataset_id_list, add_columns='PEX_SEVERE_365',
    #                    sheet_name='Sheet1')
    dataset.to_excel('out.xlsx')
