const STORAGE_KEY = 'personal_accountant_data_v2';

const DEFAULT_DATA = {
  planningMonth: { year: 1405, month: 6 },
  incomes: [
    { id: 'inc1', name: 'حقوق مرداد (برای شهریور)', amount: 64000000, forMonth: '1405/06' }
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
    { id: 'ot1', name: 'بدهی اضافی', amount: 18000000, targetMonth: '1405/06' }
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
  const [py, pm] = getPlanningMonth(appData);
  appData.incomes.push({ id: uid(), name: 'درآمد جدید', amount: 0, forMonth: `${py}/${String(pm).padStart(2, '0')}` });
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
  const [py, pm] = getPlanningMonth(appData);
  appData.oneTimeExpenses.push({ id: uid(), name: 'هزینه یک‌باره', amount: 0, targetMonth: `${py}/${String(pm).padStart(2, '0')}` });
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
      if (f.type === 'month') {
        return `<input class="date-input" type="text" value="${item[f.key] || ''}"
          onchange="updateItem('${listName}','${item.id}','${f.key}',this.value)"
          placeholder="1405/06" title="ماه مقصد">`;
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

function renderSummary(planning) {
  const cards = document.getElementById('summaryCards');
  const balanceClass = planning.available >= 0 ? 'positive' : 'negative';
  const monthLabel = `${planning.monthName} ${planning.year}`;

  cards.innerHTML = `
    <div class="card">
      <div class="card-label">درآمد ${monthLabel}</div>
      <div class="card-value income">${formatMoney(planning.totalIncome)}</div>
      <div class="card-sub">حقوق مرداد برای خرج‌های شهریور</div>
    </div>
    <div class="card">
      <div class="card-label">کل اقساط و بدهی ${monthLabel}</div>
      <div class="card-value expense">${formatMoney(planning.totalExpenses)}</div>
      <div class="card-sub">ثابت: ${formatMoney(planning.fixedTotal)} | موقت: ${formatMoney(planning.tempTotal)}${planning.oneTimeTotal ? ' | یک‌باره: ' + formatMoney(planning.oneTimeTotal) : ''}</div>
    </div>
    <div class="card">
      <div class="card-label">قابل خرج ${monthLabel}</div>
      <div class="card-value balance ${balanceClass}">${formatMoney(planning.available)}</div>
      <div class="card-sub">روزانه ~ ${formatMoney(getDailyBudget(planning.available, planning.year, planning.month))}</div>
    </div>
    <div class="card">
      <div class="card-label">درصد پوشش اقساط</div>
      <div class="card-value">${planning.totalIncome > 0 ? Math.round(planning.totalExpenses / planning.totalIncome * 100) : 0}٪</div>
      <div class="card-sub">از درآمد صرف اقساط می‌شود</div>
    </div>
  `;
}

function renderAlert(planning) {
  const box = document.getElementById('alertBox');
  const monthLabel = planning.monthName;
  if (planning.available < 0) {
    box.innerHTML = `<div class="alert danger">
      ⚠️ <strong>کسری ${formatMoney(Math.abs(planning.available))} در ${monthLabel}!</strong>
      حقوق مرداد (۶۴M) برای پوشش اقساط و بدهی‌های شهریور کافی نیست.
      ${formatMoney(Math.abs(planning.available))} باید از پس‌انداز یا جای دیگه تأمین بشه.
    </div>`;
  } else if (planning.available / planning.totalIncome < 0.15) {
    box.innerHTML = `<div class="alert warning">
      ⚡ <strong>${monthLabel} سخت!</strong> فقط ${formatMoney(planning.available)} برای کل خرج‌های زندگی داری.
      روزانه حدود ${formatMoney(getDailyBudget(planning.available, planning.year, planning.month))} — خیلی محدوده!
    </div>`;
  } else {
    box.innerHTML = `<div class="alert success">
      ✅ ${monthLabel} قابل مدیریته. ${formatMoney(planning.available)} برای خرج‌های روزمره داری
      (روزانه ~ ${formatMoney(getDailyBudget(planning.available, planning.year, planning.month))}).
    </div>`;
  }
}

function renderForecast(forecast, planning) {
  const maxVal = Math.max(...forecast.map(f => f.totalIncome), 1);

  const chart = document.getElementById('forecastChart');
  chart.innerHTML = forecast.map(f => {
    const incomeH = (f.totalIncome / maxVal) * 140;
    const expenseH = (f.totalExpenses / maxVal) * 140;
    const balanceH = (Math.max(f.available, 0) / maxVal) * 140;
    const isPlanning = f.isPlanningMonth;
    return `<div class="bar-group">
      <div class="bar-container">
        <div class="bar income-bar" style="height:${incomeH}px" title="درآمد: ${formatMoney(f.totalIncome)}"></div>
        <div class="bar expense-bar" style="height:${expenseH}px" title="اقساط: ${formatMoney(f.totalExpenses)}"></div>
        <div class="bar balance-bar" style="height:${balanceH}px" title="قابل خرج: ${formatMoney(f.available)}"></div>
      </div>
      <div class="bar-label">${f.monthName}${isPlanning ? ' ★' : ''}${f.isCurrentMonth ? ' (الان)' : ''}</div>
    </div>`;
  }).join('');

  const tbody = document.getElementById('forecastBody');
  tbody.innerHTML = forecast.map(f => {
    const status = getBudgetStatus(f.available, f.totalIncome || 1);
    const availClass = f.available >= 0 ? '' : 'style="color:var(--danger)"';
    const tags = [];
    if (f.isPlanningMonth) tags.push('بودجه فعلی');
    if (f.isCurrentMonth) tags.push('امروز');
    const tagStr = tags.length ? ` (${tags.join('، ')})` : '';
    return `<tr${f.isPlanningMonth ? ' class="planning-row"' : ''}>
      <td>${f.monthName} ${f.year}${tagStr}</td>
      <td class="amount">${formatMoney(f.totalIncome)}</td>
      <td class="amount">${formatMoney(f.totalExpenses)}</td>
      <td class="amount" ${availClass}>${formatMoney(f.available)}</td>
      <td><span class="status-badge ${status.class}">${status.label}</span></td>
    </tr>`;
  }).join('');
}

function renderBudgetBreakdown(planning) {
  const container = document.getElementById('budgetBreakdown');
  const monthLabel = `${planning.monthName} ${planning.year}`;
  const rows = planning.expenseBreakdown.map(e =>
    `<div class="budget-row">
      <span class="label">${e.name} <small>(${e.type})</small></span>
      <span class="value">${formatMoney(e.amount)}</span>
    </div>`
  ).join('');

  const expenseRatio = planning.totalIncome > 0
    ? Math.min(100, (planning.totalExpenses / planning.totalIncome) * 100)
    : 0;

  container.innerHTML = `
    <p class="budget-note">📌 بودجه ${monthLabel} — حقوق دریافتی مرداد برای خرج‌های این ماه</p>
    ${rows}
    <div class="budget-row">
      <span class="label">جمع اقساط</span>
      <span class="value">${formatMoney(planning.totalExpenses)}</span>
    </div>
    <div class="budget-row highlight">
      <span class="label">💡 بودجه پیشنهادی خرج روزمره</span>
      <span class="value ${planning.available >= 0 ? 'positive' : 'negative'}">${formatMoney(planning.available)}</span>
    </div>
    <div class="budget-row highlight">
      <span class="label">📅 بودجه روزانه (${monthLabel})</span>
      <span class="value ${planning.available >= 0 ? 'positive' : 'negative'}">${formatMoney(getDailyBudget(planning.available, planning.year, planning.month))}</span>
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
    { key: 'amount', type: 'amount' },
    { key: 'forMonth', type: 'month' }
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
    { key: 'amount', type: 'amount' },
    { key: 'targetMonth', type: 'month' }
  ]);

  const [py, pm] = getPlanningMonth(appData);
  const planning = calculateMonthBudget(appData, py, pm);
  const forecast = getForecast(appData, 6);

  renderSummary(planning);
  renderAlert(planning);
  renderForecast(forecast, planning);
  renderBudgetBreakdown(planning);
}

render();
