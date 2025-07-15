import pandas as pd

def get_columns_type(table_variables_type, df_combined):

    table_variables_type['Variable'] = table_variables_type['Variable'].str.replace('w', '{w}').str.lower()
    columns_cat = table_variables_type[table_variables_type['Type'] == 'categorical']['Variable']
    columns_ord = table_variables_type[table_variables_type['Type'] == 'ordinal']['Variable']
    columns_num = table_variables_type[table_variables_type['Type'] == 'numeric']['Variable']

    def _expand_list(lst):
        expanded = []

        for item in lst:
            if "{w}" in item:
                expanded.extend(
                    [item.replace("{w}", str(i)) for i in range(1, 6)]
                )
            else:
                expanded.append(item)

        return expanded

    columns_cat = _expand_list(columns_cat)
    columns_ord = _expand_list(columns_ord)
    columns_num = _expand_list(columns_num)

    # Remove columns_cat that are not in df_combined.columns
    columns_cat = [col for col in columns_cat if col in df_combined.columns]
    columns_ord = [col for col in columns_ord if col in df_combined.columns]
    columns_num = [col for col in columns_num if col in df_combined.columns]

    return columns_cat, columns_ord, columns_num