# ACS ECG Detector  leakage tests
def test_no_patient_leakage():
    """Проверка: train и test пациентов не пересекаются."""
    train_patients = {'P001', 'P002', 'P003'}
    test_patients = {'P004', 'P005'}
    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"Утечка! Общие пациенты: {overlap}"
