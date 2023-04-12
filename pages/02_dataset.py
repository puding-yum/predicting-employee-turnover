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
            'Usia karyawan',
            'Seberapa sering karyawan bepergian',
            'Gaji per hari',
            'Bidang pekerjaan',
            'Jarak dari rumah ke tempat kerja',
            'Tingkat pendidikan',
            'Bidang pendidikan',
            '',
            'ID karyawan',
            'Kepuasan terhadap lingkungan kerja',
            'Jenis kelamin',
            'Gaji per jam',
            'Keterlibatan kerja',
            'Tingkat kesulitan kerja',
            'Tugas dalam kerja',
            'Tingkat kepuasan terhadap pekerjaan',
            'Status pernikahan',
            'Gaji per bulan',
            'Pengeluaran internal perusahaan per bulan',
            'Jumlah perusahaan yang pernah jadi tempat kerja',
            'Usia karyawan lebih dari 18 tahun atau tidak',
            'Karyawan mengambil jam lembur atau tidak',
            'Persentase kenaikan gaji dibanding tahun lalu',
            'Performa karyawan',
            'Tingkat kepuasan hubungan karyawan dengan rekan kerja',
            'Standar jam kerja',
            'Tingkat saham',
            'Lama kerja',
            'Jumlah latihan tahun lalu',
            'Tingkat keseimbangan kerja dengan kehidupan sehari hari',
            'Lama kerja dengan perusahaan',
            'Lama kerja dengan tugas saat ini',
            'Selisih tahun sejak terakhir mendapatkan promosi',
            'Lama kerja dengan manajer saat ini',
            'Status karyawan meninggalkan perusahaan atau tidak'
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