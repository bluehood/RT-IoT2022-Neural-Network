import pandas as pd

# Arguement Parsing
import argparse
import numpy as np

def dataframeCategorySanitisation(df):
    for column in df.columns:
        # Get first element in a column
        first_entry = df[column].iloc[0]
        columns_one_host_encode = []

        # If first element is a string, use one hot encoding to vectorise the data
        if isinstance(first_entry, str):
            if column != 'Attack_type':
                columns_one_host_encode.append(column)
            else:
                df[column] = df[column].map({'MQTT_Publish':0, 'Thing_Speak': 0, 'Wipro_bulb': 0, 'DOS_SYN_Hping': 1, 'ARP_poisioning': 1, 'NMAP_UDP_SCAN': 1, 'NMAP_XMAS_TREE_SCAN': 1, 'NMAP_OS_DETECTION': 1, 'NMAP_TCP_scan': 1, 'DDOS_Slowloris': 1, 'Metasploit_Brute_Force_SSH': 1, 'NMAP_FIN_SCAN': 1})
                df = df.rename(columns={column: 'class'})
            
        
        df = pd.get_dummies(df, columns=columns_one_host_encode)
    
    return df

def dataframeNumberSanitisation(df):
    for column in df.columns:
        # Get first element in a column
        first_entry = df[column].iloc[0]

        # If first element is a float or integer, normalise data between zero and one
        if isinstance(first_entry, float) or isinstance(first_entry, int) or isinstance(first_entry, np.int64):
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df

def main():

    desc = 'Sanitise data in CSV data format before machine learning takes place.'

    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument("-i", "--input", help = "CSV formated input file.")
    parser.add_argument("-o", "--output", help = "CSV output file name.")

    # Read arguments from command line
    args = parser.parse_args()
    input_file_name = args.input
    output_file_name = args.output
    
    # Import CSV file into a pandas dataframe 
    orginal_data_df = pd.read_csv(input_file_name)
    
    # Drop uneeded columns
    orginal_data_df = orginal_data_df.drop(orginal_data_df.columns[0], axis=1)
    # print(type(orginal_data_df['src_bytes'].iloc[0]))

    # For data which is of a numerical form normalise between 0 and 1
    orginal_data_df = dataframeNumberSanitisation(orginal_data_df)

    # For each column that contains categories perform one hot encoding and save the results encoding 
    orginal_data_df = dataframeCategorySanitisation(orginal_data_df)
    
    # Replace empty values with zero
    orginal_data_df.fillna(0, inplace=True)

    print(orginal_data_df.shape[1])

    # print(orginal_data_df)
    orginal_data_df.to_csv(output_file_name, index=False)
    
    return

main()