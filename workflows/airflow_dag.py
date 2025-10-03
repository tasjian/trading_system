"""
Airflow DAG for FinRL Automatic Retraining

Schedule weekly retraining with adaptive lookback and automatic rollback.

Installation:
    pip install apache-airflow

Setup:
    1. Copy this file to your Airflow DAGs folder
    2. Update TRADING_SYSTEM_PATH to your installation path
    3. Configure schedule_interval as needed
    4. Restart Airflow scheduler

Schedule:
    - Default: Every Sunday at 2:00 AM
    - Customize with cron expression

Features:
    - Market regime detection
    - Adaptive lookback periods
    - Performance monitoring
    - Automatic rollback on degradation
    - Email notifications on failure
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys

# CONFIGURE THIS: Path to your trading system
TRADING_SYSTEM_PATH = '/Users/zac/Desktop/02_PROJECTS/01_ML4T/trading_system'

# Add trading system to path
sys.path.insert(0, TRADING_SYSTEM_PATH)

from scheduled_retraining import AutomaticRetrainingScheduler


def run_scheduled_retraining(**context):
    """Run scheduled retraining and push results to XCom."""
    scheduler = AutomaticRetrainingScheduler()
    results = scheduler.run_scheduled_retraining()

    # Push results to XCom for downstream tasks
    context['task_instance'].xcom_push(key='retraining_results', value=results)

    # Raise exception if failed
    if not results['success']:
        raise Exception(f"Retraining failed: {results.get('errors', [])}")

    return results


def check_retraining_results(**context):
    """Check retraining results and send notifications."""
    ti = context['task_instance']
    results = ti.xcom_pull(key='retraining_results', task_ids='run_retraining')

    if not results:
        print("⚠️ No results found")
        return

    print(f"\n📊 Retraining Results:")
    print(f"   Timestamp: {results['timestamp']}")
    print(f"   Success: {results['success']}")
    print(f"   Agents Retrained: {', '.join(results['agents_retrained'])}")

    if results['rollbacks']:
        print(f"   ⚠️ Rollbacks: {', '.join(results['rollbacks'])}")

    if results['errors']:
        print(f"   ❌ Errors: {', '.join(results['errors'])}")


# Default DAG args
default_args = {
    'owner': 'trading_system',
    'depends_on_past': False,
    'email': ['your-email@example.com'],  # Configure for notifications
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Create DAG
with DAG(
    'finrl_automatic_retraining',
    default_args=default_args,
    description='Automatic FinRL model retraining with adaptive lookback',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2:00 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['finrl', 'retraining', 'machine-learning'],
) as dag:

    # Task 1: Check trading system health
    health_check = BashOperator(
        task_id='health_check',
        bash_command=f'cd {TRADING_SYSTEM_PATH} && python -c "from tools.alpaca_client import alpaca_client; print(alpaca_client.get_account_info())"',
    )

    # Task 2: Run scheduled retraining
    run_retraining = PythonOperator(
        task_id='run_retraining',
        python_callable=run_scheduled_retraining,
        provide_context=True,
    )

    # Task 3: Check results
    check_results = PythonOperator(
        task_id='check_results',
        python_callable=check_retraining_results,
        provide_context=True,
    )

    # Task 4: Update trading system (reload models)
    update_system = BashOperator(
        task_id='update_system',
        bash_command=f'cd {TRADING_SYSTEM_PATH} && echo "Models updated, restart trading system if needed"',
    )

    # Define task dependencies
    health_check >> run_retraining >> check_results >> update_system


# Alternative: More frequent retraining (weekly)
with DAG(
    'finrl_weekly_retraining',
    default_args=default_args,
    description='Weekly FinRL retraining (every Sunday)',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2:00 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['finrl', 'retraining', 'weekly'],
) as weekly_dag:

    weekly_retraining = PythonOperator(
        task_id='weekly_retraining',
        python_callable=run_scheduled_retraining,
    )


# Alternative: Bi-weekly retraining (1st and 15th)
with DAG(
    'finrl_biweekly_retraining',
    default_args=default_args,
    description='Bi-weekly FinRL retraining (1st and 15th)',
    schedule_interval='0 2 1,15 * *',  # 1st and 15th at 2:00 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['finrl', 'retraining', 'biweekly'],
) as biweekly_dag:

    biweekly_retraining = PythonOperator(
        task_id='biweekly_retraining',
        python_callable=run_scheduled_retraining,
    )
