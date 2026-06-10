// JavaScript code
document.addEventListener('DOMContentLoaded', () => {
    // Render AI Detection Results
    const riskValue = document.getElementById('riskValue');


   

   



    // Edit Profile Functionality
    document.getElementById('editProfileButton').addEventListener('click', () => {
        alert('Edit Profile functionality to be implemented.');
    });

    // Refresh Results Functionality
    document.getElementById('refreshResultsButton').addEventListener('click', () => {
        alert('Refresh Results functionality to be implemented.');
    });


    // Handle form submission to save patient info and image
    document.querySelector('form').addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent default form submission
        const fileInput = document.getElementById('imageUpload');
        const resultMessage = document.getElementById('resultMessage');

        // Check if an image is uploaded
        if (fileInput.files.length === 0) {
            resultMessage.textContent = 'Please upload an image first!';
            resultMessage.style.color = 'red';
        } else {
            // Retrieve patient data from localStorage
            const patientName = localStorage.getItem("patientName");
            const patientAge = localStorage.getItem("patientAge");
            const patientGender = localStorage.getItem("patientGender");
            const patientFamilyHistory = localStorage.getItem("patientFamilyHistory");

            // Simulate transferring data to abbb.html
            localStorage.setItem("patientData", JSON.stringify({
                name: patientName,
                age: patientAge,
                gender: patientGender,
                familyHistory: patientFamilyHistory,
                image: fileInput.files[0].name // Store image name for simplicity
            }));

            resultMessage.textContent = 'Patient information and image saved successfully!';
            resultMessage.style.color = 'green';
        }
    });

    // Forgot Password Functionality
    document.getElementById('forgotPasswordLink').addEventListener('click', function() {
        document.getElementById('loginAccessRegister').classList.add('hidden');
        document.getElementById('forgotPasswordSection').classList.remove('hidden');
    });

    document.getElementById('forgotPasswordBack').addEventListener('click', function() {
        document.getElementById('forgotPasswordSection').classList.add('hidden');
        document.getElementById('loginAccessRegister').classList.remove('hidden');
    });

});

function saveAIResult(riskLevel) {
    const currentDate = new Date().toLocaleDateString(); // Get current date
    let results = JSON.parse(localStorage.getItem("healthData")) || [];

    results.push({ date: currentDate, risk: riskLevel });

    localStorage.setItem("healthData", JSON.stringify(results));
}
