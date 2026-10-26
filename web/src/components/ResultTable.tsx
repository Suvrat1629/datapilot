import { useState } from "react";
import { toCsv } from "../csv";
import { formatCell, isNumeric } from "../format";

const PAGE = 50;

interface Props {
  columns: string[];
  rows: Record<string, unknown>[];
  rowCount: number;
  truncated: boolean;
}

export default function ResultTable({ columns, rows, rowCount, truncated }: Props) {
  const [shown, setShown] = useState(PAGE);

  const download = () => {
    const blob = new Blob([toCsv(columns, rows)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "result.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card">
      <div className="card-head">
        <span>
          {rowCount.toLocaleString()} row{rowCount === 1 ? "" : "s"}
          {truncated ? " (capped)" : ""}
        </span>
        <button className="link" onClick={download}>
          Download CSV
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c} className={isNumeric(rows[0]?.[c]) ? "num" : undefined}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, shown).map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c} className={isNumeric(r[c]) ? "num" : undefined}>
                    {formatCell(r[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown < rows.length && (
        <button className="link more" onClick={() => setShown(shown + PAGE)}>
          Show {Math.min(PAGE, rows.length - shown)} more ({rows.length - shown} left)
        </button>
      )}
      {rowCount > rows.length && (
        <div className="muted small pad">Showing the first {rows.length.toLocaleString()} rows. Download has the same rows.</div>
      )}
    </div>
  );
}
