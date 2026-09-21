import mysql.connector
from mysql.connector import Error


# ──────────────────────────────────────────────
# Database Configuration
# ──────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",        # change if your MySQL user is different
    "password": "12345",            # add your MySQL password here
    "database": "lms_db"
}


# ──────────────────────────────────────────────
# Connection Helper
# ──────────────────────────────────────────────
def get_connection():
    """Returns a MySQL connection object."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"[ERROR] Could not connect to MySQL: {e}")
        return None


# ──────────────────────────────────────────────
# Generic Query Runner
# ──────────────────────────────────────────────
def run_query(sql: str, params=None):
    """
    Executes a SELECT query and returns results as a list of dicts.
    Returns an empty list on error.
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params or ())
        results = cursor.fetchall()
        return results
    except Error as e:
        print(f"[ERROR] Query failed: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────
# LMS Queries
# ──────────────────────────────────────────────

def get_all_students():
    """BR-00: Fetch all students."""
    sql = "SELECT * FROM students ORDER BY student_id ASC;"
    return run_query(sql)


def get_active_student_count():
    """BR-04: Count of active students."""
    sql = "SELECT COUNT(student_id) AS active_students FROM students WHERE status = 'Active';"
    return run_query(sql)


def get_enrollments_per_course():
    """BR-01: Total enrollments per course (most popular first)."""
    sql = """
        SELECT c.course_id, c.course_name, COUNT(e.enrollment_id) AS total_enrollments
        FROM enrollments AS e
        INNER JOIN courses AS c ON e.course_id = c.course_id
        GROUP BY c.course_id, c.course_name
        ORDER BY total_enrollments DESC;
    """
    return run_query(sql)


def get_courses_per_instructor():
    """BR-02: Number of courses each instructor is teaching."""
    sql = """
        SELECT i.instructor_id, i.name AS instructor_name,
               COUNT(c.course_id) AS total_courses_teaching
        FROM instructors i
        LEFT JOIN courses c ON i.instructor_id = c.instructor_id
        GROUP BY i.instructor_id, i.name
        ORDER BY total_courses_teaching DESC;
    """
    return run_query(sql)


def get_student_grades():
    """BR-03: Student name, course name, and total marks."""
    sql = """
        SELECT s.student_id, s.name AS student_name,
               c.course_name, g.total_marks
        FROM students AS s
        LEFT JOIN enrollments AS e  ON s.student_id   = e.student_id
        INNER JOIN grades AS g      ON e.enrollment_id = g.enrollment_id
        INNER JOIN courses AS c     ON e.course_id     = c.course_id
        ORDER BY s.student_id ASC;
    """
    return run_query(sql)


def get_enrollment_status_summary():
    """BR-05: Count of enrollments grouped by status."""
    sql = """
        SELECT status, COUNT(enrollment_id) AS total_count
        FROM enrollments
        GROUP BY status;
    """
    return run_query(sql)


def get_courses_with_no_students():
    """BR-06: Courses that have no students enrolled."""
    sql = """
        SELECT c.course_id, c.course_name, c.fee, c.status
        FROM courses c
        LEFT JOIN enrollments e ON c.course_id = e.course_id
        WHERE e.enrollment_id IS NULL;
    """
    return run_query(sql)


def get_top_students():
    """BR-07: Students ranked by highest total marks."""
    sql = """
        SELECT s.name AS student_name,
               SUM(g.total_marks) AS total_marks
        FROM students AS s
        INNER JOIN enrollments AS e ON s.student_id    = e.student_id
        INNER JOIN grades AS g      ON e.enrollment_id = g.enrollment_id
        GROUP BY s.student_id, s.name
        ORDER BY total_marks DESC;
    """
    return run_query(sql)


# ──────────────────────────────────────────────
# Quick Test (run this file directly to verify)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing database connection...\n")

    conn = get_connection()
    if conn:
        print("Connected to MySQL successfully!\n")
        conn.close()
    else:
        print("Connection failed. Check DB_CONFIG above.")
        exit(1)

    print("── All Students ──")
    for row in get_all_students():
        print(row)

    print("\n── Active Student Count ──")
    print(get_active_student_count())

    print("\n── Enrollments per Course ──")
    for row in get_enrollments_per_course():
        print(row)

    print("\n── Courses per Instructor ──")
    for row in get_courses_per_instructor():
        print(row)

    print("\n── Student Grades ──")
    for row in get_student_grades():
        print(row)

    print("\n── Enrollment Status Summary ──")
    for row in get_enrollment_status_summary():
        print(row)

    print("\n── Courses with No Students ──")
    for row in get_courses_with_no_students():
        print(row)

    print("\n── Top Students by Total Marks ──")
    for row in get_top_students():
        print(row)

