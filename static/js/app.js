document.addEventListener('DOMContentLoaded', () => {
  const input = document.querySelector('#retina-image');
  const dropZone = document.querySelector('#drop-zone');
  const preview = document.querySelector('#image-preview');
  const previewWrap = document.querySelector('#image-preview-wrap');
  const placeholder = document.querySelector('#upload-placeholder');
  const remove = document.querySelector('#remove-image');
  const form = document.querySelector('#prediction-form');
  const submit = document.querySelector('#analyze-button');
  const loading = document.querySelector('#loading-overlay');
  if (!input || !dropZone) return;
  const showFile = (file) => {
    if (!file || !['image/png', 'image/jpeg'].includes(file.type) || file.size > 10 * 1024 * 1024) { alert('Please use a PNG or JPG image smaller than 10 MB.'); return; }
    const transfer = new DataTransfer(); transfer.items.add(file); input.files = transfer.files;
    preview.src = URL.createObjectURL(file); placeholder.classList.add('d-none'); previewWrap.classList.remove('d-none'); submit.disabled = false;
  };
  dropZone.addEventListener('click', () => input.click());
  dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') input.click(); });
  input.addEventListener('change', () => showFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
  dropZone.addEventListener('drop', (event) => showFile(event.dataTransfer.files[0]));
  remove.addEventListener('click', (event) => { event.stopPropagation(); input.value = ''; preview.src = ''; previewWrap.classList.add('d-none'); placeholder.classList.remove('d-none'); submit.disabled = true; });
  form.addEventListener('submit', () => { submit.disabled = true; submit.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing'; loading.classList.remove('d-none'); });
});
