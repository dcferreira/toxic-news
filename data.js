const DataFrame = dfjs.DataFrame;

const updateTable = (table, fname) => {
    DataFrame.fromCSV(fname).then(df => {
            var data = df.toArray();
            // const columns = df.listColumns().map((item) => {
            //     return {title: item};

            table.clear();
            table.rows.add(data);
            table.draw();

        }
    );
}

const columns = [{title: "Name"}, {title: "Toxicity"},
    {title: "Severe Toxicity"}, {title: "Obscene"}, {title: "Identity Attack"},
    {title: "Insult"}, {title: "Threat"}, {title: "Sexually Explicit"}, {title: "#Days"},]

const deselectBtn = (btn) => {
    btn.classList.add("bg-gray-800", "text-gray-300");
    btn.classList.remove("bg-gray-900", "text-white", "font-bold");
}
const selectBtn = (btn) => {
    btn.classList.remove("bg-gray-800", "text-gray-300");
    btn.classList.add("bg-gray-900", "text-white", "font-bold");
}


$(document).ready(() => {
    const table = $('#example').DataTable({
        columns: columns,
        scrollX: true,
    });
    updateTable(table, "averages/all.csv");

    // buttons
    const btnAll = document.getElementById('btn-all');
    const btnYear = document.getElementById('btn-year');
    const btn90 = document.getElementById('btn-90');
    const btn30 = document.getElementById('btn-30');
    const btn7 = document.getElementById('btn-7');

    const clickButton = (name) => {
        const fname = {
            all: "averages/all.csv", year: "averages/year.csv", 90: "averages/90.csv",
            30: "averages/30.csv", 7: "averages/7.csv"
        }[name];
        const selectedBtn = {
            all: btnAll,
            year: btnYear,
            90: btn90,
            30: btn30,
            7: btn7
        }[name];

        deselectBtn(btnAll);
        deselectBtn(btnYear);
        deselectBtn(btn90);
        deselectBtn(btn30);
        deselectBtn(btn7);

        selectBtn(selectedBtn);

        updateTable(table, fname);
    }

    btnAll.addEventListener("click", () => clickButton("all"));
    btnYear.addEventListener("click", () => clickButton("year"));
    btn90.addEventListener("click", () => clickButton(90));
    btn30.addEventListener("click", () => clickButton(30));
    btn7.addEventListener("click", () => clickButton(7));
});
