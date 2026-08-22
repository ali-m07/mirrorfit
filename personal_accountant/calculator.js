const PERSIAN_MONTHS = [
  'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
  'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'
];

function formatMoney(amount) {
  const abs = Math.abs(Math.round(amount));
  return abs.toLocaleString('fa-IR') + ' تومان';
}

function parseAmount(str) {
  if (typeof str === 'number') return str;
  const cleaned = String(str).replace(/[^\d.]/g, '');
  return parseFloat(cleaned) || 0;
}

function gregorianToJalali(gy, gm, gd) {
  const g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
  let jy = gy <= 1600 ? 0 : 979;
  gy -= gy <= 1600 ? 621 : 1600;
  const gy2 = gm > 2 ? gy + 1 : gy;
  let days = 365 * gy + Math.floor((gy2 + 3) / 4) - Math.floor((gy2 + 99) / 100)
    + Math.floor((gy2 + 399) / 400) - 80 + gd + g_d_m[gm - 1];
  jy += 33 * Math.floor(days / 12053);
  days %= 12053;
  jy += 4 * Math.floor(days / 1461);
  days %= 1461;
  jy += Math.floor((days - 1) / 365);
  if (days > 365) days = (days - 1) % 365;
  const jm = days < 186 ? 1 + Math.floor(days / 31) : 7 + Math.floor((days - 186) / 30);
  const jd = 1 + (days < 186 ? days % 31 : (days - 186) % 30);
  return [jy, jm, jd];
}

function jalaliToGregorian(jy, jm, jd) {
  let gy = jy <= 979 ? 621 : 1600;
  jy -= jy <= 979 ? 0 : 979;
  const days = 365 * jy + Math.floor(jy / 33) * 8 + Math.floor((jy % 33 + 3) / 4)
    + 78 + (jm < 7 ? (jm - 1) * 31 : (jm - 7) * 30 + 186) + jd - 1;
  gy += 400 * Math.floor(days / 146097);
  let d = days % 146097;
  if (d >= 36525) {
    gy += 100 * Math.floor((d - 1) / 36525);
    d = (d - 1) % 36525;
  }
  gy += 4 * Math.floor(d / 1461);
  d %= 1461;
  if (d >= 366) {
    gy += Math.floor((d - 1) / 365);
    d = (d - 1) % 365;
  }
  const sal_a = [0, 31, (gy % 4 === 0 && gy % 100 !== 0) || gy % 400 === 0 ? 29 : 28,
    31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let gm = 0;
  while (gm < 13 && d >= sal_a[gm]) {
    d -= sal_a[gm++];
  }
  return [gy, gm, d + 1];
}

function getTodayJalali() {
  const now = new Date();
  return gregorianToJalali(now.getFullYear(), now.getMonth() + 1, now.getDate());
}

function parseJalaliDate(str) {
  const parts = String(str).replace(/-/g, '/').split('/');
  if (parts.length !== 3) return null;
  return [parseInt(parts[0], 10), parseInt(parts[1], 10), parseInt(parts[2], 10)];
}

function formatJalaliDate(jy, jm, jd) {
  return `${jy}/${String(jm).padStart(2, '0')}/${String(jd).padStart(2, '0')}`;
}

function jalaliMonthName(jm) {
  return PERSIAN_MONTHS[jm - 1] || '';
}

function isDateBeforeOrEqual(jy1, jm1, jd1, jy2, jm2, jd2) {
  if (jy1 !== jy2) return jy1 < jy2;
  if (jm1 !== jm2) return jm1 < jm2;
  return jd1 <= jd2;
}

function parseTargetMonth(str) {
  if (!str) return null;
  const parts = String(str).replace(/-/g, '/').split('/');
  if (parts.length < 2) return null;
  return [parseInt(parts[0], 10), parseInt(parts[1], 10)];
}

function matchesTargetMonth(item, targetYear, targetMonth) {
  const tm = parseTargetMonth(item.targetMonth || item.forMonth);
  if (!tm) return true;
  return tm[0] === targetYear && tm[1] === targetMonth;
}

function isTempExpenseActive(expense, targetYear, targetMonth) {
  if (!expense.endDate) return true;
  const end = parseJalaliDate(expense.endDate);
  if (!end) return true;
  const [ey, em] = end;
  if (targetYear > ey) return false;
  if (targetYear < ey) return true;
  if (targetMonth > em) return false;
  return true;
}

function isPayrollDeduction(expense) {
  return expense.deductedFromSalary !== false;
}

function getGrossIncome(data, targetYear, targetMonth) {
  const matched = data.incomes.filter(i => matchesTargetMonth(i, targetYear, targetMonth));
  if (matched.length) {
    return matched.reduce((sum, i) => sum + parseAmount(i.amount), 0);
  }
  return data.incomes.reduce((sum, i) => sum + parseAmount(i.amount), 0);
}

function calculateMonthBudget(data, targetYear, targetMonth) {
  const totalIncome = getGrossIncome(data, targetYear, targetMonth);

  const activeFixed = data.fixedExpenses.map(e => ({
    name: e.name,
    amount: parseAmount(e.amount),
    type: 'ثابت',
    deductedFromSalary: isPayrollDeduction(e)
  }));

  const activeTemp = data.tempExpenses
    .filter(e => isTempExpenseActive(e, targetYear, targetMonth))
    .map(e => ({
      name: e.name,
      amount: parseAmount(e.amount),
      type: 'موقت',
      deductedFromSalary: isPayrollDeduction(e)
    }));

  const activeOneTime = data.oneTimeExpenses
    .filter(e => matchesTargetMonth(e, targetYear, targetMonth))
    .map(e => ({
      name: e.name,
      amount: parseAmount(e.amount),
      type: 'یک‌باره',
      deductedFromSalary: isPayrollDeduction(e)
    }));

  const allExpenses = [...activeFixed, ...activeTemp, ...activeOneTime];

  const payrollDeductions = allExpenses
    .filter(e => e.deductedFromSalary)
    .reduce((sum, e) => sum + e.amount, 0);

  const shahrivarPayments = allExpenses
    .filter(e => !e.deductedFromSalary)
    .reduce((sum, e) => sum + e.amount, 0);

  const fixedTotal = activeFixed.reduce((s, e) => s + e.amount, 0);
  const tempTotal = activeTemp.reduce((s, e) => s + e.amount, 0);
  const oneTimeTotal = activeOneTime.reduce((s, e) => s + e.amount, 0);
  const totalExpenses = fixedTotal + tempTotal + oneTimeTotal;

  const netReceived = totalIncome - payrollDeductions;
  const available = netReceived - shahrivarPayments;

  const [todayY, todayM] = getTodayJalali();
  const isCurrentMonth = targetYear === todayY && targetMonth === todayM;
  const isPlanningMonth = data.planningMonth
    && data.planningMonth.year === targetYear
    && data.planningMonth.month === targetMonth;

  return {
    year: targetYear,
    month: targetMonth,
    monthName: jalaliMonthName(targetMonth),
    totalIncome,
    payrollDeductions,
    netReceived,
    shahrivarPayments,
    fixedTotal,
    tempTotal,
    oneTimeTotal,
    totalExpenses,
    available,
    isCurrentMonth,
    isPlanningMonth,
    payrollBreakdown: allExpenses.filter(e => e.deductedFromSalary),
    shahrivarBreakdown: allExpenses.filter(e => !e.deductedFromSalary),
    expenseBreakdown: allExpenses.map(e => ({
      ...e,
      type: e.deductedFromSalary ? `${e.type} · کسر از حقوق` : `${e.type} · خرج ${jalaliMonthName(targetMonth)}`
    }))
  };
}

function getPlanningMonth(data) {
  if (data.planningMonth) {
    return [data.planningMonth.year, data.planningMonth.month];
  }
  const [y, m] = getTodayJalali();
  let nm = m + 1;
  let ny = y;
  if (nm > 12) { nm = 1; ny += 1; }
  return [ny, nm];
}

function getForecast(data, monthsAhead = 6) {
  const [startY, startM] = getPlanningMonth(data);
  const forecast = [];

  for (let i = 0; i < monthsAhead; i++) {
    let y = startY;
    let m = startM + i;
    while (m > 12) {
      m -= 12;
      y += 1;
    }
    forecast.push(calculateMonthBudget(data, y, m));
  }
  return forecast;
}

function getBudgetStatus(available, totalIncome) {
  const ratio = totalIncome > 0 ? available / totalIncome : 0;
  if (available < 0) return { label: 'کسری!', class: 'bad' };
  if (ratio < 0.15) return { label: 'تنگ', class: 'tight' };
  return { label: 'مناسب', class: 'good' };
}

function getDailyBudget(available, targetYear, targetMonth) {
  const [todayY, todayM, todayD] = getTodayJalali();
  const daysInMonth = targetMonth <= 6 ? 31 : (targetMonth <= 11 ? 30 : (isLeapJalali(targetYear) ? 30 : 29));

  let remainingDays;
  if (targetYear === todayY && targetMonth === todayM) {
    remainingDays = Math.max(1, daysInMonth - todayD + 1);
  } else if (targetYear < todayY || (targetYear === todayY && targetMonth < todayM)) {
    remainingDays = 1;
  } else {
    remainingDays = daysInMonth;
  }
  return available / remainingDays;
}

function isLeapJalali(jy) {
  const breaks = [1, 5, 9, 13, 17, 22, 26, 30];
  const jp = jy - (jy >= 0 ? 474 : 473);
  const jy2 = 474 + (jp % 2820);
  return breaks.includes((jy2 + 38) * 682 % 2816);
}
