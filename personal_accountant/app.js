const STORAGE_KEY = 'personal_accountant_data';

const DEFAULT_DATA = {
  incomes: [
    { id: 'inc1', name: 'حقوق ماهانه', amount: 64000000 }
  ],
  fixedExpenses: [
    { id: 'fix1', name: 'دیجی‌پی', amount: 9500000 },
    { id: 'fix2', name: 'اسنپ', amount: 6000000 },
    { id: 'fix3', name: 'وام ملت', amount: 7100000 }
  ],
  tempExpenses: [
    { id: 'tmp1', name: 'رسالت', amount: 15000000, endDate: '1405/07/29' },
    { id: 'tmp2', name: 'قسط شهریور', amount: 2100000, endDate: '1405/06/05' },
    { id: 'tmp3', name: 'قسط آذر', amount: 2800000, endDate: '1405/09/10' },
    { id: 'tmp4', name: 'قسط شرکت', amount: 6500000, endDate: '1405/12/29' }
  ],
  oneTimeExpenses: [
    { id: 'ot1', name: 'بدهی اضافی این ماه', amount: 18000000 }
  ]
};

let appData = loadData();

function loadData() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch (_) { /* ignore */ }
  return JSON.parse(JSON.stringify(DEFAULT_DATA));
}

function saveData() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(appData));
  render();
}

function uid() {
  return 'id_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
}

function addIncome() {
  appData.incomes.push({ id: uid(), name: 'درآمد جدید', amount: 0 });
  saveData();
}

function addFixedExpense() {
  appData.fixedExpenses.push({ id: uid(), name: 'قسط جدید', amount: 0 });
  saveData();
}

function addTempExpense() {
  appData.tempExpenses.push({ id: uid(), name: 'قسط موقت', amount: 0, endDate: '1405/12/29' });
  saveData();
}

function addOneTimeExpense() {
  appData.oneTimeExpenses.push({ id: uid(), name: 'هزینه یک‌باره', amount: 0 });
  saveData();
}

function deleteItem(list, id) {
  appData[list] = appData[list].filter(item => item.id !== id);
  saveData();
}

function updateItem(list, id, field, value) {
  const item = appData[list].find(i => i.id === id);
  if (item) {
    item[field] = value;
    saveData();
  }
}

function renderItemList(containerId, items, listName, fields) {
  const container = document.getElementById(containerId);
  container.innerHTML = items.map(item => {
    const inputs = fields.map(f => {
      if (f.type === 'amount') {
        return `<input class="amount ${listName === 'incomes' ? 'income-color' : ''}" type="text"
          value="${parseAmount(item[f.key]).toLocaleString('en-US')}"
          onchange="updateItem('${listName}','${item.id}','${f.key}',parseAmount(this.value)); this.value=parseAmount(this.value).toLocaleString('en-US')"
          placeholder="مبلغ">`;
      }
      if (f.type === 'date') {
        return `<input class="date-input" type="text" value="${item[f.key] || ''}"
          onchange="updateItem('${listName}','${item.id}','${f.key}',this.value)"
          placeholder="1405/07/29">`;
      }
      return `<input type="text" value="${item[f.key]}"
        onchange="updateItem('${listName}','${item.id}','${f.key}',this.value)"
        placeholder="نام">`;
    }).join('');

    return `<div class="item-row">
      ${inputs}
      <button class="btn-delete" onclick="deleteItem('${listName}','${item.id}')" title="حذف">✕</button>
    </div>`;
  }).join('');
}

function renderSummary(current) {
  const cards = document.getElementById('summaryCards');
  const balanceClass = current.available >= 0 ? 'positive' : 'negative';

  cards.innerHTML = `
    <div class="card">
      <div class="card-label">درآمد این ماه</div>
      <div class="card-value income">${formatMoney(current.totalIncome)}</div>
    </div>
    <div class="card">
      <div class="card-label">کل اقساط و بدهی</div>
      <div class="card-value expense">${formatMoney(current.totalExpenses)}</div>
      <div class="card-sub">ثابت: ${formatMoney(current.fixedTotal)} | موقت: ${formatMoney(current.tempTotal)}${current.oneTimeTotal ? ' | یک‌باره: ' + formatMoney(current.oneTimeTotal) : ''}</div>
    </div>
    <div class="card">
      <div class="card-label">قابل خرج این ماه</div>
      <div class="card-value balance ${balanceClass}">${formatMoney(current.available)}</div>
      <div class="card-sub">روزانه ~ ${formatMoney(getDailyBudget(current.available))}</div>
    </div>
    <div class="card">
      <div class="card-label">درصد پوشش اقساط</div>
      <div class="card-value">${current.totalIncome > 0 ? Math.round(current.totalExpenses / current.totalIncome * 100) : 0}٪</div>
      <div class="card-sub">از درآمد صرف اقساط می‌شود</div>
    </div>
  `;
}

function renderAlert(current) {
  const box = document.getElementById('alertBox');
  if (current.available < 0) {
    box.innerHTML = `<div class="alert danger">
      ⚠️ <strong>کسری ${formatMoney(Math.abs(current.available))}!</strong>
      اقساط و بدهی‌های این ماه از درآمدت بیشتره. باید ${formatMoney(Math.abs(current.available))} از پس‌انداز یا جای دیگه تأمین کنی.
    </div>`;
  } else if (current.available / current.totalIncome < 0.15) {
    box.innerHTML = `<div class="alert warning">
      ⚡ <strong>ماه سخت!</strong> فقط ${formatMoney(current.available)} برای کل خرج‌های زندگی (خوراک، حمل‌ونقل، تفریح...) داری.
      روزانه حدود ${formatMoney(getDailyBudget(current.available))} — خیلی محدوده!
    </div>`;
  } else {
    box.innerHTML = `<div class="alert success">
      ✅ وضعیت مالی این ماه قابل مدیریته. ${formatMoney(current.available)} برای خرج‌های روزمره داری
      (روزانه ~ ${formatMoney(getDailyBudget(current.available))}).
    </div>`;
  }
}

function renderForecast(forecast) {
  const maxVal = Math.max(...forecast.map(f => f.totalIncome), 1);

  const chart = document.getElementById('forecastChart');
  chart.innerHTML = forecast.map(f => {
    const incomeH = (f.totalIncome / maxVal) * 140;
    const expenseH = (f.totalExpenses / maxVal) * 140;
    const balanceH = (Math.max(f.available, 0) / maxVal) * 140;
    return `<div class="bar-group">
      <div class="bar-container">
        <div class="bar income-bar" style="height:${incomeH}px" title="درآمد: ${formatMoney(f.totalIncome)}"></div>
        <div class="bar expense-bar" style="height:${expenseH}px" title="اقساط: ${formatMoney(f.totalExpenses)}"></div>
        <div class="bar balance-bar" style="height:${balanceH}px" title="قابل خرج: ${formatMoney(f.available)}"></div>
      </div>
      <div class="bar-label">${f.monthName}${f.isCurrentMonth ? ' ★' : ''}</div>
    </div>`;
  }).join('');

  const tbody = document.getElementById('forecastBody');
  tbody.innerHTML = forecast.map(f => {
    const status = getBudgetStatus(f.available, f.totalIncome);
    const availClass = f.available >= 0 ? '' : 'style="color:var(--danger)"';
    return `<tr>
      <td>${f.monthName} ${f.year}${f.isCurrentMonth ? ' (الان)' : ''}</td>
      <td class="amount">${formatMoney(f.totalIncome)}</td>
      <td class="amount">${formatMoney(f.totalExpenses)}</td>
      <td class="amount" ${availClass}>${formatMoney(f.available)}</td>
      <td><span class="status-badge ${status.class}">${status.label}</span></td>
    </tr>`;
  }).join('');
}

function renderBudgetBreakdown(current) {
  const container = document.getElementById('budgetBreakdown');
  const rows = current.expenseBreakdown.map(e =>
    `<div class="budget-row">
      <span class="label">${e.name} <small>(${e.type})</small></span>
      <span class="value">${formatMoney(e.amount)}</span>
    </div>`
  ).join('');

  const expenseRatio = current.totalIncome > 0
    ? Math.min(100, (current.totalExpenses / current.totalIncome) * 100)
    : 0;

  container.innerHTML = `
    ${rows}
    <div class="budget-row">
      <span class="label">جمع اقساط</span>
      <span class="value">${formatMoney(current.totalExpenses)}</span>
    </div>
    <div class="budget-row highlight">
      <span class="label">💡 بودجه پیشنهادی خرج روزمره</span>
      <span class="value ${current.available >= 0 ? 'positive' : 'negative'}">${formatMoney(current.available)}</span>
    </div>
    <div class="budget-row highlight">
      <span class="label">📅 بودجه روزانه (تا آخر ماه)</span>
      <span class="value ${current.available >= 0 ? 'positive' : 'negative'}">${formatMoney(getDailyBudget(current.available))}</span>
    </div>
    <div class="progress-bar-wrap">
      <div class="progress-label">
        <span>سهم اقساط از درآمد</span>
        <span>${Math.round(expenseRatio)}٪</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:${expenseRatio}%; background:${expenseRatio > 90 ? 'var(--danger)' : expenseRatio > 70 ? 'var(--warning)' : 'var(--success)'}"></div>
      </div>
    </div>
  `;
}

function renderDate() {
  const [jy, jm, jd] = getTodayJalali();
  document.getElementById('currentDate').textContent =
    `امروز: ${jd} ${jalaliMonthName(jm)} ${jy}`;
}

function render() {
  renderDate();

  renderItemList('incomeList', appData.incomes, 'incomes', [
    { key: 'name', type: 'text' },
    { key: 'amount', type: 'amount' }
  ]);
  renderItemList('fixedExpenseList', appData.fixedExpenses, 'fixedExpenses', [
    { key: 'name', type: 'text' },
    { key: 'amount', type: 'amount' }
  ]);
  renderItemList('tempExpenseList', appData.tempExpenses, 'tempExpenses', [
    { key: 'name', type: 'text' },
    { key: 'amount', type: 'amount' },
    { key: 'endDate', type: 'date' }
  ]);
  renderItemList('oneTimeList', appData.oneTimeExpenses, 'oneTimeExpenses', [
    { key: 'name', type: 'text' },
    { key: 'amount', type: 'amount' }
  ]);

  const [cy, cm] = getTodayJalali();
  const current = calculateMonthBudget(appData, cy, cm);
  const forecast = getForecast(appData, 6);

  renderSummary(current);
  renderAlert(current);
  renderForecast(forecast);
  renderBudgetBreakdown(current);
}

render();
