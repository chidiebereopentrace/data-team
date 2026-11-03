---
status: accepted
---

# A weekly run reports coverage and failed refs to BigQuery

The OTA job is a daily collect-and-export. Its history is files on disk, and BigQuery loading is a separate first load that will not accept a second day. An earlier map sent the human summary by email and treated that mail as the notification.

The ingestion engine runs once a week. It reads every manifest CSV in a directory. `source_id` is unique across those files. After the loads, it appends three tables in the report dataset: one row for the run, coverage counts of prepared rows by indicator class, country, and year, and one row for each manifest row that did not fully load. The report is written before the process exits. A failed or partial ref makes the exit non-zero and does not erase the other rows or the report.

Email is not the record of the run. The report tables are.
