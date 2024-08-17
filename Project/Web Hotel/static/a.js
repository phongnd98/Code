function clickOpen(divId) {
    var div = document.getElementById(divId);
    if (div.style.display==='none') {
        div.style.display='block';
    } else {
        div.style.display='none';
    }
}


document.addEventListener("DOMContentLoaded", function() {
    clickOpen('chat')
});