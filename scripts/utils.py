from  sklearn.preprocessing import OrdinalEncoder


def processNa(df):
    "suppression des colonnes en gardant celles avec moins de 2000 NA puis suppression des lignes avec NA restantes"
    df.dropna(axis=1,thresh = (df.shape[0])-2000, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df

def encodage(df):
    "Transformation des variables qualitatives en quantitatives"
    df_after_enc = df.copy(deep=True)
    obj_col=df_after_enc.select_dtypes('object').columns
    enc = OrdinalEncoder()
    enc_data= enc.fit_transform(df_after_enc[obj_col])
    df[obj_col]=enc_data
    df[obj_col]=df[obj_col].astype("int64")

    return df

def renameTarget(df):
    df.rename(columns={"correct_fedas_code": "Target"}, inplace =True)
    return df