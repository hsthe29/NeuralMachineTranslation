let max_history_size = 10;
let l = 0;
function addRow(src_text, tar_text) {
    const div = document.getElementById('history');

    l += 1;
    if (l > 10) {
        div.removeChild(div.lastChild);
    }

    let subdiv = document.createElement("div");
    subdiv.classList.add("history-row");
    let src = document.createElement("textarea");
    src.value = src_text;
    let tar = document.createElement("textarea");
    tar.value = tar_text;
    subdiv.appendChild(src);
    subdiv.appendChild(tar);
    div.insertBefore(subdiv, div.firstChild);
    let max_height = Math.max(src.scrollHeight, tar.scrollHeight);
    src.style.height = "0px";
    tar.style.height = "0px";
    src.style.height = max_height + "px";
    tar.style.height = max_height + "px";
}
function clearTable() {
    table = document.getElementById("history-table")
    while(table.rows.length > 1) {
        table.deleteRow(1)
    }
}

function postData() {
    let src_text = document.getElementById('src_text').value;
    let target_cell = document.getElementById('tar_text');
    let data = {
        lang: 'en',
        text: src_text
    }
    target_cell.value = "...";
    fetch('', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    }).then(response => response.text())
        .then(data => {
            let tar_text = JSON.parse(data).text
            document.getElementById('tar_text').value = tar_text
            addRow(src_text, tar_text)
        })
        .catch(function(error) {
                console.error(error);
    });
}

function clearAll() {
    let textarea = document.getElementsByClassName("autoresizing");
    Array.prototype.forEach.call(textarea, function (text) {
        text.value = "";
        text.style.height = 'auto';
    });
}

function showToast() {
    var toast = document.getElementById("toast");
    toast.classList.add("show");
    setTimeout(function(){ toast.classList.remove("show"); }, 3000);
}

function shutDown(element) {
    element.classList.add("unclickable");
    element.style.backgroundColor = 'yellow';
    element.style.width = '15px';
    element.style.height = '15px';
    element.title = "Shutting down";

    const spinElement = document.createElement('div');
    spinElement.classList.add('spin-circle');
    element.appendChild(spinElement);

    let  data = "shutdown"

    fetch('', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json'
        },
        body: data
    }).then(response => response.text())
        .then(data => {
            tar_text = JSON.parse(data).status
            if (tar_text === "shutdown") {
                element.style.borderRadius = '0';
                element.style.backgroundColor = 'red';
                const spinElement = element.querySelector('.spin-circle');
                element.removeChild(spinElement);
                element.title = "Disconnected";
                currentState = 'square';
            }
        })
        .catch(function(error) {
                console.error(error);
    })
    .finally(() => {
            showToast();
        }
    )
}

let textarea = document.getElementsByClassName("autoresizing");
Array.prototype.forEach.call(textarea, text =>
    text.addEventListener('input', autoResize, false)
);

function autoResize() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight  - 20 + 'px';
}

function showHistory() {
    let element = document.getElementById('history');

    if (element.style.height === "0px" || element.style.height === "") {
        const height = element.scrollHeight;
        // element.style.height = `${height}px`;
        element.style.height = "100%";
    } else {
        element.style.height = "0px";
    }
}