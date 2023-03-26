import streamlit as st
import pandas as pd

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Dataset",
    layout="centered",
    page_icon="random",
)

st.markdown("<h1 style='text-align: center'>Dataset</h1>", unsafe_allow_html=True)
file_name = None
if 'file_name' not in st.session_state:
    uploaded_file = st.file_uploader("Upload dataset")
    if uploaded_file is not None:
        st.session_state['file_name'] = uploaded_file.name
        file_name = uploaded_file.name
else:
    file_name = st.session_state['file_name']

if file_name is not None:
    
    if file_name == "employee.csv":
        dataset = pd.read_csv("data/employee_class_right.csv")

        unique_value = dataset.nunique()

        description = [
            "Umur",                                                                                                 #Age
            "Frekuensi travel karyawan",                                                                            #BusinessTravel
            "Gaji per hari",                                                                                        #DailyRate
            "Departemen",                                                                                           #Department
            "Jarak dari rumah",                                                                                     #DistanceFromHome
            "Pendidikan (1=dibawah SMA, 2=SMA, 3=S1, 4=S2, 5=S3)",                                                  #Education
            "Bidang pendidikan (1=HR, 2=LIFE SCIENCES, 3=MARKETING, 4=MEDICAL SCIENCES, 5=OTHERS, 6=TEHCNICAL)",    #EducationField
            "",                                                                                                     #EmployeeCount                                 
            "Nomor karyawan",                                                                                       #EmployeeNumber
            "Kepuasan terhadap lingungan kerja",                                                                    #EnvironmentSatisfaction
            "Jenis kelamin",                                                                                        #Gender
            "Gaji per jam",                                                                                         #HourlyRate
            "Keterlibatan kerja (1=Low, 2=Medium, 3=High, 4=Very High)",                                            #JobInvolvement
            "Tingkat kesulitan kerja",                                                                              #JobLevel
            "Tugas dalam kerja",                                                                                    #JobRole
            "Tingkat kepuasan terhadap pekerjaan (1=Low, 2=Medium, 3=High, 4=Very High)",                           #JobSatisfaction
            "Status pernikahan (1=pisah, 2=menikah, 3=single)",                                                     #MaritalStatus
            "Gaji per bulan",                                                                                       #MonthlyIncome
            "Pengeluaran internal per bulan",                                                                       #MonthlyRate
            "Jumlah perusahaan yang pernah jadi tempat kerja",                                                      #NumCompaniesWorked
            "Usia lebih dari 18 tahun",                                                                             #Over18
            "Lembur",                                                                                               #OverTime
            "Persentase kenaikan gaji dibanding tahun lalu",                                                        #PercentSalaryHike
            "Rating performa (1=Low, 2=Good, 3=Excellent, 4=Outstanding)",                                          #PerformanceRating
            "Kepuasan hubungan dengan rekan kerja (1=Low, 2=Medium, 3=High, 4=Very High)",                          #RelationshipSatisfaction
            "Standar jam kerja",                                                                                    #StandardHours
            "Level saham",                                                                                          #StockOptionLevel
            "Lama kerja",                                                                                           #TotalWorkingYears
            "Jumlah latihan tahun lalu",                                                                            #TrainingTimesLastYear
            "Tingkat keseimbangan kerja dengan kehidupan sehari hari (1=Bad, 2=Good, 3=Better, 4=Best)",            #WorkLifeBalance
            "Lama kerja dengan perusahaan",                                                                         #YearsAtCompany
            "Lama kerja dengan tugas saat ini",                                                                     #YearsInCurrentRole
            "Selisih tahun sejak terakhir mendapatkan promosi",                                                     #YearsSinceLastPromotion
            "Lama kerja dengan manajer",                                                                            #YearsWithCurrManager
            "Status karyawan meninggalkan perusahaan atau tidak",                                                   #Attrition
        ]
        data_type = [
            "numerik",          #Age
            "kategorikal",      #BusinessTravel
            "numerik",          #DailyRate
            "kategorikal",      #Department
            "numerik",          #DistanceFromHome
            "kategorikal",      #Education
            "kategorikal",      #EducationField
            "kategorikal",      #EmployeeCount
            "numerik",          #EmployeeNumber
            "kategorikal",      #EnvironmentSatisfaction
            "kategorikal",      #Gender
            "numerik",          #HourlyRate
            "kategorikal",      #JobInvolvement
            "kategorikal",      #JobLevel
            "kategorikal",      #JobRole
            "kategorikal",      #JobSatisfaction
            "kategorikal",      #MaritalStatus
            "numerik",          #MonthlyIncome
            "numerik",          #MonthlyRate
            "numerik",          #NumCompaniesWorked
            "kategorikal",      #Over18
            "kategorikal",      #OverTime
            "numerik",          #PercentSalaryHike
            "kategorikal",      #PerformanceRating
            "kategorikal",      #RelationshipSatisfaction
            "numerik",          #StandardHours
            "kategorikal",      #StockOptionLevel
            "numerik",          #TotalWorkingYears
            "numerik",          #TrainingTimesLastYear
            "kategorikal",      #WorkLifeBalance
            "numerik",          #YearsAtCompany
            "numerik",          #YearsInCurrentRole
            "numerik",          #YearsSinceLastPromotion
            "numerik",          #YearsWithCurrManager
            "kategorikal",      #Attrition
        ] 

        # print(len(unique_value))
        # print(len(description))
        # print(len(data_type))
        # print(len(dataset.columns))

        data_description = pd.DataFrame({"Nama Fitur":dataset.columns, "Nilai Unik":unique_value, "Tipe Data":data_type, "Keterangan":description })
        data_description = data_description.reset_index(drop=True)

        tab1, tab2 = st.tabs(["Dataset", "Deskripsi Fitur"])
        with tab1:
            st.dataframe(dataset)

        with tab2:
            st.dataframe(data_description)

        st.write("Dataset merupakan data tentang employee turnover, dataset memiliki 35 fitur dan 1470 baris data. Tiap baris data merupakan representasi dari seorang karyawan. Terdapat 2 kelas dalam fitur Attrition yaitu yes dan no.", unsafe_allow_html=True)
    else:
        dataset = pd.read_csv("data/"+file_name)
        st.dataframe(dataset)
        
    st.session_state['dataset']=dataset