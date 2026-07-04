import demoCsv from "./assets/demo.csv?raw";
import demoWorkbookUrl from "./assets/demo_workbook.xlsx?url";

export function demoCsvFile(): File {
  return new File([demoCsv], "demo_sales_report.csv", { type: "text/csv" });
}

export async function demoWorkbookFile(): Promise<File> {
  const blob = await (await fetch(demoWorkbookUrl)).blob();
  return new File([blob], "demo_q2_workbook.xlsx", {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}
