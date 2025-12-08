// static/js/main.js

// Весь код — после загрузки DOM
document.addEventListener("DOMContentLoaded", function () {
    // 1. Автоматическое скрытие flash-сообщений через несколько секунд
    const alerts = document.querySelectorAll(".alert");
    alerts.forEach((alertEl) => {
        // Оставляем время, чтобы пользователь успел прочитать
        setTimeout(() => {
            // bootstrap.Alert доступен из подключённого bundle
            if (window.bootstrap && alertEl.classList.contains("show")) {
                const alert = bootstrap.Alert.getOrCreateInstance(alertEl);
                alert.close();
            } else {
                // На всякий случай, если bootstrap не подгрузился
                alertEl.style.display = "none";
            }
        }, 6000);
    });

    // 2. Активация tooltip'ов Bootstrap (если где-то добавишь data-bs-toggle="tooltip")
    if (window.bootstrap) {
        const tooltipTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="tooltip"]')
        );
        tooltipTriggerList.forEach(function (tooltipTriggerEl) {
            new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // 3. Лёгкая валидация всех форм (HTML5 checkValidity + подсветка Bootstrap)
    const forms = document.querySelectorAll("form");
    forms.forEach((form) => {
        form.addEventListener(
            "submit",
            function (event) {
                // Если форма невалидна — предотвращаем отправку и включаем подсветку
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add("was-validated");
            },
            false
        );
    });

    // 4. Небольшой помощник для поля выбора файла (CSV)
    const csvInput = document.getElementById("csvFile");
    if (csvInput) {
        csvInput.addEventListener("change", function () {
            const fileName = csvInput.files && csvInput.files.length > 0
                ? csvInput.files[0].name
                : "Файл не выбран";

            // Ищем соседний .form-text и дописываем туда имя файла
            const wrapper = csvInput.closest(".mb-3") || csvInput.parentElement;
            if (!wrapper) return;

            let info = wrapper.querySelector(".csv-file-name-info");
            if (!info) {
                info = document.createElement("div");
                info.className = "form-text csv-file-name-info";
                wrapper.appendChild(info);
            }
            info.textContent = "Выбран файл: " + fileName;
        });
    }
});
