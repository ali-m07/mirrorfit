const STORAGE_KEY = 'personal_accountant_data_v4';

const DEFAULT_DATA = {
  planningMonth: { year: 1405, month: 6 },
  incomes: [
    { id: 'inc1', name: 'حقوق مرداد (ناخالص)', amount: 64000000, forMonth: '1405/06' }
  ],
  fixedExpenses: [
    { id: 'fix1', name: 'دیجی‌پی', amount: 9500000, deductedFromSalary: true },
    { id: 'fix2', name: 'اسنپ', amount: 6000000, deductedFromSalary: true },
    { id: 'fix3', name: 'وام ملت', amount: 7100000, deductedFromSalary: true }
  ],
  tempExpenses: [
    { id: 'tmp1', name: 'رسالت', amount: 15000000, endDate: '1405/07/29', deductedFromSalary: true },
    { id: 'tmp2', name: 'قسط شهریور', amount: 2100000, endDate: '1405/06/05', deductedFromSalary: true },
    { id: 'tmp3', name: 'قسط آذر', amount: 2800000, endDate: '1405/09/10', deductedFromSalary: true },
    { id: 'tmp4', name: 'قسط شرکت', amount: 6500000, endDate: '1405/12/29', deductedFromSalary: false }
  ],
  oneTimeExpenses: [
    { id: 'ot1', name: 'بدهی اضافی', amount: 18000000, targetMonth: '1405/06', deductedFromSalary: true }
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
  appData.fixedExpenses.push({ id: uid(), name: 'قسط جدید', amount: 0, deductedFromSalary: true });
  saveData();
}

function addTempExpense() {
  appData.tempExpenses.push({ id: uid(), name: 'قسط موقت', amount: 0, endDate: '1405/12/29', deductedFromSalary: true });
  saveData();
}

function addOneTimeExpense() {
  const [py, pm] = getPlanningMonth(appData);
  appData.oneTimeExpenses.push({
    id: uid(), name: 'هزینه یک‌باره', amount: 0,
    targetMonth: `${py}/${String(pm).padStart(2, '0')}`,
    deductedFromSalary: false
  });
  saveData();
}

function deleteItem(list, id) {
  appData[list] = appData[list].filter(item => item.id !== id);
  saveData();
}

function updateItem(list, id, field, value) {
  const item = appData[list].find(i => i.id === id);
  if (item) {
    item[field] = field === 'deductedFromSalary' ? !!value : value;
    saveData();
  }
}

function togglePayroll(list, id) {
  const item = appData[list].find(i => i.id === id);
  if (item) {
    item.deductedFromSalary = !isPayrollDeduction(item);
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
      if (f.type === 'payroll') {
        const checked = isPayrollDeduction(item);
        return `<label class="payroll-toggle" title="کسر از حقوق مرداد">
          <input type="checkbox" ${checked ? 'checked' : ''} onchange="togglePayroll('${listName}','${item.id}')">
          <span>کسر از حقوق</span>
        </label>`;
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

function renderSummary(p) {
  const cards = document.getElementById('summaryCards');
  const balanceClass = p.available >= 0 ? 'positive' : 'negative';

  cards.innerHTML = `
    <div class="card">
      <div class="card-label">حقوق مرداد (ناخالص)</div>
      <div class="card-value income">${formatMoney(p.totalIncome)}</div>
      <div class="card-sub">دریافتی آخر مرداد</div>
    </div>
    <div class="card">
      <div class="card-label">کسر شده از حقوق</div>
      <div class="card-value expense">${formatMoney(p.payrollDeductions)}</div>
      <div class="card-sub">اقساطی که قبل از واریز کم شد</div>
    </div>
    <div class="card">
      <div class="card-label">واریز خالص (برای شهریور)</div>
      <div class="card-value balance">${formatMoney(p.netReceived)}</div>
      <div class="card-sub">مبلغی که به حسابت رسید</div>
    </div>
    <div class="card">
      <div class="card-label">مانده برای زندگی در شهریور</div>
      <div class="card-value balance ${balanceClass}">${formatMoney(p.available)}</div>
      <div class="card-sub">بعد از خرج‌های شهریور · روزانه ~ ${formatMoney(getDailyBudget(p.available, p.year, p.month))}</div>
    </div>
  `;
}

function renderAlert(p) {
  const box = document.getElementById('alertBox');
  if (p.available < 0) {
    box.innerHTML = `<div class="alert danger">
      ⚠️ <strong>${formatMoney(Math.abs(p.available))} کسری برای زندگی در شهریور!</strong><br>
      حقوق مرداد: ${formatMoney(p.totalIncome)} − کسر اقساط: ${formatMoney(p.payrollDeductions)} = واریز ${formatMoney(p.netReceived)}<br>
      خرج‌های شهریور: ${formatMoney(p.shahrivarPayments)} → ${formatMoney(Math.abs(p.available))} کم داری.
    </div>`;
  } else if (p.available / p.netReceived < 0.15) {
    box.innerHTML = `<div class="alert warning">
      ⚡ <strong>شهریور سخت!</strong> از ${formatMoney(p.netReceived)} واریزی، ${formatMoney(p.shahrivarPayments)} خرج شهریور داری.
      فقط ${formatMoney(p.available)} برای زندگی (خوراک، حمل‌ونقل...) — روزانه ~ ${formatMoney(getDailyBudget(p.available, p.year, p.month))}
    </div>`;
  } else {
    box.innerHTML = `<div class="alert success">
      ✅ ${formatMoney(p.available)} برای زندگی در شهریور داری
      (واریز ${formatMoney(p.netReceived)} − خرج شهریور ${formatMoney(p.shahrivarPayments)}).
      روزانه ~ ${formatMoney(getDailyBudget(p.available, p.year, p.month))}
    </div>`;
  }
}

function renderForecast(forecast) {
  const maxVal = Math.max(...forecast.map(f => f.netReceived || f.totalIncome), 1);

  const chart = document.getElementById('forecastChart');
  chart.innerHTML = forecast.map(f => {
    const netH = ((f.netReceived || 0) / maxVal) * 140;
    const extraH = ((f.shahrivarPayments || 0) / maxVal) * 140;
    const balanceH = (Math.max(f.available, 0) / maxVal) * 140;
    return `<div class="bar-group">
      <div class="bar-container">
        <div class="bar income-bar" style="height:${netH}px" title="واریز خالص: ${formatMoney(f.netReceived)}"></div>
        <div class="bar expense-bar" style="height:${extraH}px" title="خرج ماه: ${formatMoney(f.shahrivarPayments)}"></div>
        <div class="bar balance-bar" style="height:${balanceH}px" title="برای زندگی: ${formatMoney(f.available)}"></div>
      </div>
      <div class="bar-label">${f.monthName}${f.isPlanningMonth ? ' ★' : ''}</div>
    </div>`;
  }).join('');

  const tbody = document.getElementById('forecastBody');
  tbody.innerHTML = forecast.map(f => {
    const status = getBudgetStatus(f.available, f.netReceived || 1);
    const availClass = f.available >= 0 ? '' : 'style="color:var(--danger)"';
    return `<tr${f.isPlanningMonth ? ' class="planning-row"' : ''}>
      <td>${f.monthName} ${f.year}${f.isPlanningMonth ? ' (شهریور)' : ''}</td>
      <td class="amount">${formatMoney(f.netReceived)}</td>
      <td class="amount">${formatMoney(f.shahrivarPayments)}</td>
      <td class="amount" ${availClass}>${formatMoney(f.available)}</td>
      <td><span class="status-badge ${status.class}">${status.label}</span></td>
    </tr>`;
  }).join('');
}

function renderBudgetBreakdown(p) {
  const container = document.getElementById('budgetBreakdown');

  const payrollRows = p.payrollBreakdown.map(e =>
    `<div class="budget-row deduct">
      <span class="label">− ${e.name} <small>(کسر از حقوق)</small></span>
      <span class="value expense-val">${formatMoney(e.amount)}</span>
    </div>`
  ).join('');

  const shahrivarRows = p.shahrivarBreakdown.map(e =>
    `<div class="budget-row deduct">
      <span class="label">− ${e.name} <small>(خرج شهریور)</small></span>
      <span class="value expense-val">${formatMoney(e.amount)}</span>
    </div>`
  ).join('');

  container.innerHTML = `
    <p class="budget-note">📌 حقوق مرداد می‌رسه → اقساط کسر می‌شه → باقی‌مونده رو تو <strong>شهریور</strong> باهاش زندگی می‌کنی</p>

    <div class="flow-section">
      <div class="budget-row">
        <span class="label">حقوق مرداد (ناخالص)</span>
        <span class="value positive">${formatMoney(p.totalIncome)}</span>
      </div>
      ${payrollRows}
      <div class="budget-row subtotal">
        <span class="label">= واریز خالص به حساب</span>
        <span class="value">${formatMoney(p.netReceived)}</span>
      </div>
    </div>

    <div class="flow-section">
      <p class="flow-title">خرج‌های شهریور (از واریزی)</p>
      ${shahrivarRows || '<p class="empty-note">خرج اضافه‌ای ثبت نشده</p>'}
      <div class="budget-row subtotal">
        <span class="label">= جمع خرج شهریور</span>
        <span class="value expense-val">${formatMoney(p.shahrivarPayments)}</span>
      </div>
    </div>

    <div class="budget-row highlight final">
      <span class="label">🎯 مانده برای زندگی در شهریور</span>
      <span class="value ${p.available >= 0 ? 'positive' : 'negative'}">${formatMoney(p.available)}</span>
    </div>
    <div class="budget-row highlight">
      <span class="label">📅 بودجه روزانه شهریور</span>
      <span class="value ${p.available >= 0 ? 'positive' : 'negative'}">${formatMoney(getDailyBudget(p.available, p.year, p.month))}</span>
    </div>
  `;
}

function renderDate() {
  const [jy, jm, jd] = getTodayJalali();
  document.getElementById('currentDate').textContent =
    `امروز: ${jd} ${jalaliMonthName(jm)} ${jy} · بودجه: شهریور`;
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
    { key: 'amount', type: 'amount' },
    { key: 'deductedFromSalary', type: 'payroll' }
  ]);
  renderItemList('tempExpenseList', appData.tempExpenses, 'tempExpenses', [
    { key: 'name', type: 'text' },
    { key: 'amount', type: 'amount' },
    { key: 'endDate', type: 'date' },
    { key: 'deductedFromSalary', type: 'payroll' }
  ]);
  renderItemList('oneTimeList', appData.oneTimeExpenses, 'oneTimeExpenses', [
    { key: 'name', type: 'text' },
    { key: 'amount', type: 'amount' },
    { key: 'targetMonth', type: 'month' },
    { key: 'deductedFromSalary', type: 'payroll' }
  ]);

  const [py, pm] = getPlanningMonth(appData);
  const planning = calculateMonthBudget(appData, py, pm);
  const forecast = getForecast(appData, 6);

  renderSummary(planning);
  renderAlert(planning);
  renderForecast(forecast);
  renderBudgetBreakdown(planning);
}

render();
